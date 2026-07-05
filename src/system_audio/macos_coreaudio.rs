//! macOS Core Audio Process Tap 实现（macOS >= 14.4）。
//!
//! 通过 `AudioHardwareCreateProcessTap` 创建全系统音频 tap，
//! 创建私有 aggregate device 组合默认输出与 tap，IOProc 回调中
//! downmix → resample → 16kHz mono f32 → mpsc channel。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use super::watchdog::{is_all_zero, SilenceWatchdog};
use super::{
    DROPPED_FRAME_WARN_EVERY, TAP_REBUILD_SILENCE_MS, TAP_SAMPLE_CHANNEL_CAPACITY,
    TARGET_SAMPLE_RATE,
};

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use anyhow::{anyhow, bail, Context, Result};
use core_foundation::{
    array::CFArray, base::TCFType, boolean::CFBoolean, dictionary::CFDictionary, string::CFString,
};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_audio::{
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceMainSubDeviceKey,
    kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey,
    kAudioDevicePropertyDeviceNameCFString, kAudioDevicePropertyDeviceUID,
    kAudioDevicePropertyStreamConfiguration, kAudioHardwarePropertyDefaultOutputDevice,
    kAudioHardwarePropertyDevices, kAudioHardwarePropertyTranslatePIDToProcessObject,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeOutput, kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
    kAudioTapPropertyFormat, AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID,
    AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyProcessTap, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize,
    AudioObjectID, AudioObjectPropertyAddress, CATapDescription, CATapMuteBehavior,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatLinearPCM, AudioBuffer, AudioBufferList,
    AudioStreamBasicDescription, AudioTimeStamp,
};
use objc2_foundation::{NSArray, NSNumber, NSProcessInfo, NSString};
use tracing::{info, warn};

const K_AUDIO_OBJECT_SYSTEM_OBJECT: AudioObjectID = 1;
const K_CAPTURE_AGGREGATE_UID: &str = "com.arcvoice.capture";
const K_CAPTURE_AGGREGATE_NAME: &str = "ArcVoice Capture";
const OS_STATUS_OK: i32 = 0;

const MIN_PROCESS_TAP_MACOS_MAJOR: i64 = 14;
const MIN_PROCESS_TAP_MACOS_MINOR: i64 = 4;

type CFStringRef = *const c_void;
type CFDictionaryRef = *const c_void;

unsafe extern "C" {
    fn AudioHardwareCreateAggregateDevice(
        in_description: CFDictionaryRef,
        out_device_id: *mut AudioObjectID,
    ) -> i32;
    fn AudioHardwareDestroyAggregateDevice(in_device_id: AudioObjectID) -> i32;
}

/// macOS osstatus 转可读字符串（FourCC 或数字）。
fn osstatus_to_string(status: i32) -> String {
    let bytes = status.to_be_bytes();
    let fourcc = bytes
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
        .then(|| String::from_utf8_lossy(&bytes).into_owned());
    match fourcc {
        Some(code) => format!("{status} ('{code}')"),
        None => status.to_string(),
    }
}

pub(super) fn is_process_tap_supported() -> bool {
    let version = NSProcessInfo::processInfo().operatingSystemVersion();
    (version.majorVersion as i64, version.minorVersion as i64)
        >= (MIN_PROCESS_TAP_MACOS_MAJOR, MIN_PROCESS_TAP_MACOS_MINOR)
}

pub(super) struct TapObjects {
    aggregate_id: AudioObjectID,
    tap_id: AudioObjectID,
    io_proc_id: AudioDeviceIOProcID,
    client_data: NonNull<TapCallbackState>,
    started: bool,
}

struct TapCallbackState {
    tx: mpsc::Sender<Vec<f32>>,
    session_stop_flag: Arc<AtomicBool>,
    callback_stop_flag: Arc<AtomicBool>,
    needs_rebuild: Arc<AtomicBool>,
    in_rate: u32,
    channels: usize,
    format_flags: u32,
    bytes_per_frame: u32,
    #[allow(dead_code)]
    frames_per_packet: u32,
    bits_per_channel: u32,
    layout_logged: AtomicBool,
    watchdog: SilenceWatchdog,
    dropped_frames: AtomicU64,
}

pub(super) unsafe fn create_tap_objects(
    stop_flag: Arc<AtomicBool>,
    needs_rebuild: Arc<AtomicBool>,
    device_uid: Option<&str>,
) -> Result<(TapObjects, mpsc::Receiver<Vec<f32>>)> {
    let (tx, rx) = mpsc::channel(TAP_SAMPLE_CHANNEL_CAPACITY);
    remove_existing_capture_aggregate_devices();

    let process_object_id = translate_current_pid_to_process_object()?;
    let tap_description = create_tap_description(process_object_id);

    let mut tap_id: AudioObjectID = 0;
    let status = unsafe { AudioHardwareCreateProcessTap(Some(&tap_description), &mut tap_id) };
    ensure_status(status, "AudioHardwareCreateProcessTap")?;

    let result = unsafe { create_aggregate_for_tap(&tap_description, tap_id, device_uid) };
    let aggregate_id = match result {
        Ok(id) => id,
        Err(err) => {
            destroy_process_tap(tap_id);
            return Err(err);
        }
    };

    let format = match unsafe { read_tap_format(tap_id) } {
        Ok(format) => format,
        Err(err) => {
            unsafe {
                let _ = AudioHardwareDestroyAggregateDevice(aggregate_id);
            }
            destroy_process_tap(tap_id);
            return Err(err);
        }
    };
    let in_rate = format.mSampleRate.round() as u32;
    let channels = format.mChannelsPerFrame.max(1) as usize;
    info!(
        tap_id,
        aggregate_id, in_rate, channels, "CoreAudio process tap created"
    );

    let callback_state = Box::new(TapCallbackState {
        tx,
        session_stop_flag: stop_flag,
        callback_stop_flag: Arc::new(AtomicBool::new(false)),
        needs_rebuild,
        in_rate,
        channels,
        format_flags: format.mFormatFlags,
        bytes_per_frame: format.mBytesPerFrame,
        frames_per_packet: format.mFramesPerPacket,
        bits_per_channel: format.mBitsPerChannel,
        layout_logged: AtomicBool::new(false),
        watchdog: SilenceWatchdog::new(TAP_REBUILD_SILENCE_MS),
        dropped_frames: AtomicU64::new(0),
    });
    let client_data = NonNull::new(Box::into_raw(callback_state))
        .ok_or_else(|| anyhow!("failed to allocate tap callback state"))?;

    let mut io_proc_id: AudioDeviceIOProcID = Some(tap_io_proc);
    let io_proc_ptr = NonNull::new(&mut io_proc_id as *mut AudioDeviceIOProcID)
        .ok_or_else(|| anyhow!("failed to prepare IOProc pointer"))?;
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            aggregate_id,
            Some(tap_io_proc),
            client_data.as_ptr().cast::<c_void>(),
            io_proc_ptr,
        )
    };
    if let Err(err) = ensure_status(status, "AudioDeviceCreateIOProcID") {
        unsafe {
            drop(Box::from_raw(client_data.as_ptr()));
            let _ = AudioHardwareDestroyAggregateDevice(aggregate_id);
        }
        destroy_process_tap(tap_id);
        return Err(err);
    }

    let status = unsafe { AudioDeviceStart(aggregate_id, io_proc_id) };
    if let Err(err) = ensure_status(status, "AudioDeviceStart") {
        unsafe {
            let _ = AudioDeviceDestroyIOProcID(aggregate_id, io_proc_id);
            drop(Box::from_raw(client_data.as_ptr()));
            let _ = AudioHardwareDestroyAggregateDevice(aggregate_id);
        }
        destroy_process_tap(tap_id);
        return Err(err);
    }

    Ok((
        TapObjects {
            aggregate_id,
            tap_id,
            io_proc_id,
            client_data,
            started: true,
        },
        rx,
    ))
}

pub(super) unsafe fn destroy_tap_objects(objects: TapObjects) {
    mark_callback_state_closing_ptr(objects.client_data);

    if objects.started {
        let status = unsafe { AudioDeviceStop(objects.aggregate_id, objects.io_proc_id) };
        warn_status(status, "AudioDeviceStop");
    }

    let status = unsafe { AudioDeviceDestroyIOProcID(objects.aggregate_id, objects.io_proc_id) };
    warn_status(status, "AudioDeviceDestroyIOProcID");

    // CoreAudio can still deliver a tail IOProc callback shortly after Stop/Destroy.
    // Keep client_data alive for process lifetime to avoid use-after-free.
    unsafe { std::mem::forget(Box::from_raw(objects.client_data.as_ptr())) };

    let status = unsafe { AudioHardwareDestroyAggregateDevice(objects.aggregate_id) };
    warn_status(status, "AudioHardwareDestroyAggregateDevice");

    destroy_process_tap(objects.tap_id);
}

fn mark_callback_state_closing_ptr(client_data: NonNull<TapCallbackState>) {
    let state = unsafe { client_data.as_ref() };
    state.callback_stop_flag.store(true, Ordering::Release);
}

unsafe extern "C-unwind" fn tap_io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input_data: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output_data: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client_data: *mut c_void,
) -> i32 {
    if client_data.is_null() {
        return OS_STATUS_OK;
    }

    let state = unsafe { &mut *(client_data.cast::<TapCallbackState>()) };
    if state.callback_stop_flag.load(Ordering::Acquire)
        || state.session_stop_flag.load(Ordering::Acquire)
    {
        return OS_STATUS_OK;
    }

    let buffers = unsafe { input_data.as_ref() };
    if buffers.mNumberBuffers == 0 {
        return OS_STATUS_OK;
    }

    if !state.layout_logged.swap(true, Ordering::AcqRel) {
        let buffer_ptr = buffers.mBuffers.as_ptr();
        let mut descs = Vec::new();
        for index in 0..(buffers.mNumberBuffers as usize).min(8) {
            let buffer = unsafe { &*buffer_ptr.add(index) };
            let sample_count = buffer.mDataByteSize as usize / std::mem::size_of::<f32>();
            let all_zero = if buffer.mData.is_null() || sample_count == 0 {
                true
            } else {
                let samples =
                    unsafe { std::slice::from_raw_parts(buffer.mData.cast::<f32>(), sample_count) };
                is_all_zero(samples)
            };
            descs.push(format!(
                "#{index}:ch={} bytes={} samples={} zero={}",
                buffer.mNumberChannels, buffer.mDataByteSize, sample_count, all_zero
            ));
        }
        info!(
            buffers = buffers.mNumberBuffers,
            tap_channels = state.channels,
            in_rate = state.in_rate,
            format_flags = state.format_flags,
            bytes_per_frame = state.bytes_per_frame,
            bits_per_channel = state.bits_per_channel,
            detail = %descs.join("; "),
            "CoreAudio tap buffer layout"
        );
    }

    let audio_buffers = unsafe {
        std::slice::from_raw_parts(buffers.mBuffers.as_ptr(), buffers.mNumberBuffers as usize)
    };
    let Some((mono, all_zero)) = (unsafe { downmix_audio_buffers(audio_buffers, state.channels) })
    else {
        return OS_STATUS_OK;
    };
    let frame_count = mono.len();
    let frame_ms = ((frame_count as u64 * 1_000) / u64::from(state.in_rate.max(1))).max(1);
    if state.watchdog.observe(all_zero, frame_ms) {
        state.needs_rebuild.store(true, Ordering::Release);
    }

    let pcm16k = resample_linear(&mono, state.in_rate, TARGET_SAMPLE_RATE);
    if state.tx.try_send(pcm16k).is_err() {
        let dropped = state.dropped_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped % DROPPED_FRAME_WARN_EVERY == 0 {
            warn!(
                dropped_frames = dropped,
                "system audio tap channel full; dropping frame"
            );
        }
    }

    OS_STATUS_OK
}

unsafe fn create_tap_description(process_object_id: AudioObjectID) -> Retained<CATapDescription> {
    let excluded_pid = NSNumber::new_u32(process_object_id);
    let excluded = NSArray::from_retained_slice(&[excluded_pid]);
    let description = unsafe {
        CATapDescription::initStereoGlobalTapButExcludeProcesses(
            CATapDescription::alloc(),
            &excluded,
        )
    };
    unsafe {
        description.setName(&NSString::from_str("ArcVoice System Capture"));
        description.setMuteBehavior(CATapMuteBehavior::Unmuted);
        description.setPrivate(true);
    }
    description
}

unsafe fn create_aggregate_for_tap(
    tap_description: &CATapDescription,
    tap_id: AudioObjectID,
    device_uid: Option<&str>,
) -> Result<AudioObjectID> {
    // 选定输出设备：指定 UID 或默认输出。
    let output_uid = match device_uid {
        Some(uid) => uid.to_string(),
        None => {
            let default_id = read_u32_property(
                K_AUDIO_OBJECT_SYSTEM_OBJECT,
                kAudioHardwarePropertyDefaultOutputDevice,
                kAudioObjectPropertyScopeGlobal,
                None,
            )
            .context("failed to read default output device")?;
            read_cf_string_property(
                default_id,
                kAudioDevicePropertyDeviceUID,
                kAudioObjectPropertyScopeGlobal,
            )
            .context("failed to read default output UID")?
        }
    };
    info!(output_uid = %output_uid, "creating aggregate for tap");
    let tap_uuid = unsafe { tap_description.UUID() }.UUIDString().to_string();

    let tap_dict = CFDictionary::from_CFType_pairs(&[
        (
            CFString::new(cstr_key(kAudioSubTapUIDKey)).into_CFType(),
            CFString::new(&tap_uuid).into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioSubTapDriftCompensationKey)).into_CFType(),
            CFBoolean::true_value().into_CFType(),
        ),
    ]);
    let tap_list = CFArray::from_CFTypes(&[tap_dict]);

    let dict = CFDictionary::from_CFType_pairs(&[
        (
            CFString::new(cstr_key(kAudioAggregateDeviceNameKey)).into_CFType(),
            CFString::new(K_CAPTURE_AGGREGATE_NAME).into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioAggregateDeviceUIDKey)).into_CFType(),
            CFString::new(K_CAPTURE_AGGREGATE_UID).into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioAggregateDeviceIsPrivateKey)).into_CFType(),
            CFBoolean::true_value().into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioAggregateDeviceMainSubDeviceKey)).into_CFType(),
            CFString::new(&output_uid).into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioAggregateDeviceTapListKey)).into_CFType(),
            tap_list.into_CFType(),
        ),
        (
            CFString::new(cstr_key(kAudioAggregateDeviceTapAutoStartKey)).into_CFType(),
            CFBoolean::true_value().into_CFType(),
        ),
    ]);

    let mut aggregate_id: AudioObjectID = 0;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(
            dict.as_concrete_TypeRef() as CFDictionaryRef,
            &mut aggregate_id,
        )
    };
    if let Err(err) = ensure_status(status, "AudioHardwareCreateAggregateDevice") {
        warn!(tap_id, error = %err, "failed to create aggregate for process tap");
        return Err(err);
    }

    Ok(aggregate_id)
}

unsafe fn translate_current_pid_to_process_object() -> Result<AudioObjectID> {
    let pid = std::process::id() as i32;
    let qualifier_size = std::mem::size_of::<i32>() as u32;
    read_u32_property(
        K_AUDIO_OBJECT_SYSTEM_OBJECT,
        kAudioHardwarePropertyTranslatePIDToProcessObject,
        kAudioObjectPropertyScopeGlobal,
        Some(((&pid as *const i32).cast::<c_void>(), qualifier_size)),
    )
    .with_context(|| format!("failed to translate pid {pid} to CoreAudio process object"))
}

unsafe fn read_tap_format(tap_id: AudioObjectID) -> Result<AudioStreamBasicDescription> {
    let mut format: AudioStreamBasicDescription = unsafe { std::mem::zeroed() };
    let mut data_size = std::mem::size_of::<AudioStreamBasicDescription>() as u32;
    let mut address = property_address(kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal);
    let status = unsafe {
        AudioObjectGetPropertyData(
            tap_id,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
            NonNull::new((&mut format as *mut AudioStreamBasicDescription).cast::<c_void>())
                .ok_or_else(|| anyhow!("null buffer"))?,
        )
    };
    ensure_status(status, "AudioObjectGetPropertyData(tap format)")?;

    if format.mSampleRate <= 0.0 || format.mChannelsPerFrame == 0 {
        bail!("tap returned invalid format: {format:?}");
    }
    if format.mFormatID != kAudioFormatLinearPCM
        || (format.mFormatFlags & kAudioFormatFlagIsFloat) == 0
        || format.mBitsPerChannel != 32
    {
        bail!("tap format is not f32 linear PCM: {format:?}");
    }

    Ok(format)
}

unsafe fn read_u32_property(
    object_id: AudioObjectID,
    selector: u32,
    scope: u32,
    qualifier: Option<(*const c_void, u32)>,
) -> Result<u32> {
    let mut value: u32 = 0;
    let mut data_size = std::mem::size_of::<u32>() as u32;
    let mut address = property_address(selector, scope);
    let (qualifier_ptr, qualifier_size) = qualifier.unwrap_or((ptr::null(), 0));
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            NonNull::from(&mut address),
            qualifier_size,
            qualifier_ptr,
            NonNull::from(&mut data_size),
            NonNull::new((&mut value as *mut u32).cast::<c_void>())
                .ok_or_else(|| anyhow!("null buffer"))?,
        )
    };
    ensure_status(status, "AudioObjectGetPropertyData(u32)")?;
    Ok(value)
}

unsafe fn read_cf_string_property(
    object_id: AudioObjectID,
    selector: u32,
    scope: u32,
) -> Result<String> {
    let mut value: CFStringRef = ptr::null();
    let mut data_size = std::mem::size_of::<CFStringRef>() as u32;
    let mut address = property_address(selector, scope);
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
            NonNull::new((&mut value as *mut CFStringRef).cast::<c_void>())
                .ok_or_else(|| anyhow!("null buffer"))?,
        )
    };
    ensure_status(status, "AudioObjectGetPropertyData(CFString)")?;
    if value.is_null() {
        bail!("CoreAudio returned null CFString");
    }
    let cf_string = unsafe { CFString::wrap_under_get_rule(value.cast()) };
    Ok(cf_string.to_string())
}

/// 查询设备在某 scope（Output / Input）下的缓冲区数量。
/// 用于判断设备方向：Output scope 下 mNumberBuffers>0 才是真正的输出设备。
///
/// 读 `kAudioDevicePropertyStreamConfiguration`，它返回变长 `AudioBufferList`，
/// 但我们只需要第一个字段 `mNumberBuffers`，所以只看头部即可。
unsafe fn device_buffer_count(object_id: AudioObjectID, scope: u32) -> Result<u32> {
    let mut address = property_address(kAudioDevicePropertyStreamConfiguration, scope);
    let mut data_size: u32 = 0;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            object_id,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
        )
    };
    ensure_status(
        status,
        "AudioObjectGetPropertyDataSize(StreamConfiguration)",
    )?;
    if data_size < std::mem::size_of::<u32>() as u32 {
        return Ok(0);
    }
    let mut buf = vec![0u8; data_size as usize];
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
            NonNull::new(buf.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| anyhow!("null buffer"))?,
        )
    };
    ensure_status(status, "AudioObjectGetPropertyData(StreamConfiguration)")?;
    Ok(unsafe { ptr::read_unaligned(buf.as_ptr().cast::<u32>()) })
}

/// 列出所有**输出方向**的音频设备（id, name, uid）。
///
/// 按 Output scope 的 `StreamConfiguration` 过滤：只有该 scope 下
/// `mNumberBuffers > 0` 的设备才保留。这样会剔除纯输入设备（麦克风），
/// 也会把同名设备里的「输入引擎」条目（如 USB 耳机的 input 子项）滤掉，
/// 只留真正的输出设备。
pub fn enumerate_output_devices() -> Result<Vec<super::OutputDevice>> {
    let device_ids = unsafe { get_all_device_ids() }?;
    let mut devices = Vec::new();
    for &device_id in &device_ids {
        // 先判方向：不是输出设备就跳过。
        let out_buffers =
            unsafe { device_buffer_count(device_id, kAudioObjectPropertyScopeOutput) }.unwrap_or(0);
        if out_buffers == 0 {
            continue;
        }
        let uid = match unsafe {
            read_cf_string_property(
                device_id,
                kAudioDevicePropertyDeviceUID,
                kAudioObjectPropertyScopeGlobal,
            )
        } {
            Ok(uid) => uid,
            Err(_) => continue,
        };
        let name = match unsafe {
            read_cf_string_property(
                device_id,
                kAudioDevicePropertyDeviceNameCFString,
                kAudioObjectPropertyScopeGlobal,
            )
        } {
            Ok(name) => name,
            Err(_) => continue,
        };
        devices.push(super::OutputDevice {
            id: device_id,
            name,
            uid,
        });
    }
    Ok(devices)
}

unsafe fn get_all_device_ids() -> Result<Vec<AudioObjectID>> {
    let mut address = property_address(
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
    );
    let mut data_size: u32 = 0;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            K_AUDIO_OBJECT_SYSTEM_OBJECT,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
        )
    };
    ensure_status(status, "AudioObjectGetPropertyDataSize(devices)")?;

    let count = data_size as usize / std::mem::size_of::<AudioObjectID>();
    let mut devices = vec![0_u32; count];
    let status = unsafe {
        AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM_OBJECT,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut data_size),
            NonNull::new(devices.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| anyhow!("null buffer"))?,
        )
    };
    ensure_status(status, "AudioObjectGetPropertyData(devices)")?;
    Ok(devices)
}

unsafe fn remove_existing_capture_aggregate_devices() {
    let devices = match unsafe { get_all_device_ids() } {
        Ok(devices) => devices,
        Err(err) => {
            warn!(error = %err, "failed to enumerate devices for aggregate cleanup");
            return;
        }
    };

    for device_id in devices {
        let should_remove = unsafe {
            read_cf_string_property(
                device_id,
                kAudioDevicePropertyDeviceUID,
                kAudioObjectPropertyScopeGlobal,
            )
        }
        .map(|uid| uid == K_CAPTURE_AGGREGATE_UID)
        .unwrap_or(false);

        if should_remove {
            let status = unsafe { AudioHardwareDestroyAggregateDevice(device_id) };
            warn_status(status, "AudioHardwareDestroyAggregateDevice(existing)");
        }
    }
}

fn property_address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn cstr_key(key: &std::ffi::CStr) -> &str {
    key.to_str().unwrap_or_default()
}

fn ensure_status(status: i32, operation: &str) -> Result<()> {
    if status == OS_STATUS_OK {
        Ok(())
    } else {
        Err(anyhow!(
            "{operation} failed: osstatus={}",
            osstatus_to_string(status)
        ))
    }
}

fn warn_status(status: i32, operation: &str) {
    if status != OS_STATUS_OK {
        warn!(operation, osstatus = %osstatus_to_string(status), "CoreAudio cleanup non-zero status");
    }
}

fn destroy_process_tap(tap_id: AudioObjectID) {
    let status = unsafe { AudioHardwareDestroyProcessTap(tap_id) };
    warn_status(status, "AudioHardwareDestroyProcessTap");
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    let channels = channels.max(1);
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

unsafe fn downmix_audio_buffers(
    buffers: &[AudioBuffer],
    fallback_channels: usize,
) -> Option<(Vec<f32>, bool)> {
    let mut views = Vec::with_capacity(buffers.len());

    for buffer in buffers {
        if buffer.mData.is_null() || buffer.mDataByteSize == 0 {
            continue;
        }

        let sample_count = buffer.mDataByteSize as usize / std::mem::size_of::<f32>();
        if sample_count == 0 {
            continue;
        }

        let samples =
            unsafe { std::slice::from_raw_parts(buffer.mData.cast::<f32>(), sample_count) };
        let channels = if buffer.mNumberChannels > 0 {
            buffer.mNumberChannels as usize
        } else if buffers.len() == 1 {
            fallback_channels.max(1)
        } else {
            1
        };
        let frame_count = sample_count / channels;
        if frame_count > 0 {
            views.push((samples, channels, frame_count));
        }
    }

    if views.is_empty() {
        return None;
    }

    let all_zero = views.iter().all(|(samples, _, _)| is_all_zero(samples));
    if views.len() == 1 {
        let (samples, channels, _) = views[0];
        return Some((downmix_to_mono(samples, channels), all_zero));
    }

    let frame_count = views
        .iter()
        .map(|(_, _, frame_count)| *frame_count)
        .min()
        .unwrap_or(0);
    if frame_count == 0 {
        return None;
    }

    let total_channels: usize = views.iter().map(|(_, channels, _)| *channels).sum();
    let mut mono = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let mut sum = 0.0;
        for (samples, channels, _) in &views {
            let start = frame_index * *channels;
            for sample in &samples[start..start + *channels] {
                sum += *sample;
            }
        }
        mono.push(sum / total_channels as f32);
    }

    Some((mono, all_zero))
}

fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let dst_len = ((samples.len() as f64) * dst_rate as f64 / src_rate as f64).round() as usize;
    let mut out = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let pos = i as f64 * src_rate as f64 / dst_rate as f64;
        let left = pos.floor() as usize;
        let frac = (pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }

    out
}
