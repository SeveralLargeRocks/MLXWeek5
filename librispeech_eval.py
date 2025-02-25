import torch
import data_stream
import utils
import whisper
import jiwer

def main(): 

    device = utils.device

    model = whisper.load_model("tiny.en", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    model.eval()
    torch.set_grad_enabled(False)

    test_stream = data_stream.WhisperPreprocessor(
        data_stream.load_dataset(split="test", config_name="clean"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_stream, 
        batch_size=4, 
        collate_fn=data_stream.collate_fn
    )
    
    all_predictions = []
    all_references = []

    # Process each batch and calculate metrics
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Optional: limit number of evaluation samples
            break

    # Process each sample in the batch
        for j in range(batch["audio"].shape[0]):
            # Get single audio sample
            audio = batch["audio"][j].numpy()
            
            # Get model prediction
            result = model.transcribe(audio)
            prediction = result["text"]
            
            # Get ground truth
            reference = tokenizer.decode(batch["tokens"][j].tolist())
            
            # Store results for WER calculation
            all_predictions.append(prediction)
            all_references.append(reference)
            
            # Print individual sample results
            print(f"Sample {i*4+j+1}:")
            print(f"  Prediction: {prediction}")
            print(f"  Reference:  {reference}")
        
        # For each batch, transcribe a sample and print the ground truth and prediction
        result = model.transcribe(batch["audio"][0].numpy())
        print("Prediction: ", result["text"][0])
        print("Ground truth: ", tokenizer.decode(batch["tokens"][0].tolist()))

    # Calculate WER using jiwer
    error_rate = jiwer.wer(all_references, all_predictions)
    print(f"\nWord Error Rate (WER): {error_rate:.4f}")


if __name__ == "__main__":
    main()
        
        
        