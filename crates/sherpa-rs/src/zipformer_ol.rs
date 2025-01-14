use crate::{
    get_default_provider,
    utils::{cstr_to_string, RawCStr},
};
use eyre::{bail, Result};
use std::{mem, slice};
use tracing::{debug, info};

#[derive(Debug, Default)]
pub struct ZipFormerOnlineConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,

    pub num_threads: Option<i32>,
    pub provider: Option<String>,
    pub debug: bool,
    pub long_decode: bool,
}

pub struct ZipFormerOnline {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: Option<ZipFormerStream>,
}

impl ZipFormerOnline {
    pub fn new(config: ZipFormerOnlineConfig) -> Result<Self> {
        // Zipformer config
        let decoder_ptr = RawCStr::new(&config.decoder);
        let encoder_ptr = RawCStr::new(&config.encoder);
        let joiner_ptr = RawCStr::new(&config.joiner);
        let provider_ptr = RawCStr::new(&config.provider.unwrap_or(get_default_provider()));
        let tokens_ptr = RawCStr::new(&config.tokens);
        let decoding_method_ptr = RawCStr::new("greedy_search");

        let transcuder_config = sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            joiner: joiner_ptr.as_ptr(),
        };
        // Offline model config
        let mut model_config: sherpa_rs_sys::SherpaOnnxOnlineModelConfig = unsafe { mem::zeroed() };
        model_config.num_threads = config.num_threads.unwrap_or(1);
        model_config.debug = config.debug.into();
        model_config.provider = provider_ptr.as_ptr();
        model_config.transducer = transcuder_config;
        model_config.tokens = tokens_ptr.as_ptr();

        // Recognizer config
        let mut recognizer_config: sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig =
            unsafe { mem::zeroed() };
        recognizer_config.model_config = model_config;
        recognizer_config.decoding_method = decoding_method_ptr.as_ptr();

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer")
        }
        let mut my = Self {
            recognizer,
            stream: None,
        };
        if config.long_decode {
            my.stream = Some(ZipFormerStream::new(recognizer)?);
        }
        Ok(my)
    }

    pub fn decode(&mut self, sample_rate: u32, samples: &[f32]) -> Result<ZipFormerResult> {
        match self.stream {
            Some(ref mut stream) => stream.decode(self.recognizer, sample_rate, samples),
            None => {
                let mut stream = ZipFormerStream::new(self.recognizer)?;
                stream.decode(self.recognizer, sample_rate, samples)
            }
        }
    }

    pub fn reset_decode(&mut self) {
        if let Some(ref mut stream) = self.stream {
            stream.reset_decode(self.recognizer);
        }
    }

    pub fn is_endpoint(&self) -> bool {
        if let Some(ref stream) = self.stream {
            stream.is_endpoint(self.recognizer)
        } else {
            true
        }
    }

    pub fn finish_input(&mut self) {
        if let Some(ref mut stream) = self.stream {
            stream.finish();
        }
    }
}

unsafe impl Send for ZipFormerOnline {}
unsafe impl Sync for ZipFormerOnline {}

impl Drop for ZipFormerOnline {
    fn drop(&mut self) {
        info!("drop ZipFormerOnline {:p}", self.recognizer);
        drop(self.stream.take());
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

struct ZipFormerStream(*const sherpa_rs_sys::SherpaOnnxOnlineStream);

impl ZipFormerStream {
    fn new(recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer) -> Result<Self> {
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            bail!("create stream fail");
        }
        Ok(Self(stream))
    }
    fn decode(
        &mut self,
        recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
        sample_rate: u32,
        samples: &[f32],
    ) -> Result<ZipFormerResult> {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.0,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(recognizer, self.0) > 0 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(recognizer, self.0);
            }
            let result = sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(recognizer, self.0);
            // let result = sherpa_rs_sys::SherpaOnnxGetOnlineStreamResultAsJson(recognizer, self.0);

            if result.is_null() {
                bail!("get result failed");
            }
            let raw_result = result.read();
            debug!("result: {:?}", raw_result);
            let res = raw_result.into();
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result);

            // let text = cstr_to_string(result);
            // sherpa_rs_sys::SherpaOnnxDestroyOnlineStreamResultJson(result);
            // Ok(text)
            Ok(res)
        }
    }

    pub fn reset_decode(&mut self, recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer) {
        unsafe { sherpa_rs_sys::SherpaOnnxOnlineStreamReset(recognizer, self.0) }
    }
    pub fn is_endpoint(
        &self,
        recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    ) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(recognizer, self.0) == 1 }
    }
    pub fn finish(&mut self) {
        unsafe { sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.0) }
    }
}

impl Drop for ZipFormerStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            info!("drop ZipFormerStream {:p}", self.0);
            unsafe {
                sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.0);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZipFormerResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
}

impl From<sherpa_rs_sys::SherpaOnnxOnlineRecognizerResult> for ZipFormerResult {
    fn from(value: sherpa_rs_sys::SherpaOnnxOnlineRecognizerResult) -> Self {
        if value.count > 0 {
            unsafe {
                let timestamps = slice::from_raw_parts(value.timestamps, value.count as usize);
                let c_tokens = slice::from_raw_parts(value.tokens_arr, value.count as usize);
                let tokens: Vec<String> = c_tokens.iter().map(|v| cstr_to_string(*v)).collect();
                Self {
                    text: cstr_to_string(value.text),
                    tokens,
                    timestamps: Vec::from(timestamps),
                }
            }
        } else {
            Self {
                text: "".to_string(),
                tokens: vec![],
                timestamps: vec![],
            }
        }
    }
}

impl ZipFormerResult {
    pub fn segments(&self, split_sec: f32) -> Vec<String> {
        let mut last_index = 0;
        let mut result = Vec::new();
        for i in 1..self.timestamps.len() {
            if self.timestamps[i] - self.timestamps[i - 1] >= split_sec {
                let seg = (&self.tokens[last_index..i]).join("");
                result.push(seg);
                last_index = i;
            }
        }
        if last_index != self.timestamps.len() - 1 {
            let seg = (&self.tokens[last_index..self.timestamps.len()]).join("");
            result.push(seg);
        }
        result
    }
}
