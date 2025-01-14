/*
Use ASR models for extract text from audio

Chinese:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2

cargo run --example zipformer_ol -- \
    "sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/test_wavs/DEV_T0000000000.wav" \
    "sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/encoder-epoch-20-avg-1-chunk-16-left-128.onnx" \
    "sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/decoder-epoch-20-avg-1-chunk-16-left-128.onnx" \
    "sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/joiner-epoch-20-avg-1-chunk-16-left-128.onnx" \
    "sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/tokens.txt"
*/

use sherpa_rs::zipformer_ol::ZipFormerOnline;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (wav_path, encoder_path, decoder_path, joiner_path, tokens_path) = (
        args.get(1).expect("Missing wav file path argument"),
        args.get(2).expect("Missing encoder path argument"),
        args.get(3).expect("Missing decoder path argument"),
        args.get(4).expect("Missing joiner path argument"),
        args.get(5).expect("Missing tokens path argument"),
    );

    // Read the WAV file
    let (samples, sample_rate) = sherpa_rs::read_audio_file(wav_path).unwrap();

    let config = sherpa_rs::zipformer_ol::ZipFormerOnlineConfig {
        encoder: encoder_path.into(),
        decoder: decoder_path.into(),
        joiner: joiner_path.into(),
        tokens: tokens_path.into(),
        // long_decode: true,
        ..Default::default()
    };
    let mut zipformer = ZipFormerOnline::new(config).unwrap();
    for i in 0..10 {
        let text = zipformer.decode(sample_rate, samples.clone()).unwrap();
        println!("✅{} Text: {}", i, text);
        // zipformer.reset_decode();
    }
}
