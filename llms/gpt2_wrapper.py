import os
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List

logger = logging.getLogger(__name__)

class GPT2Wrapper:
    """
    Wrapper class for GPT-2 model that implements LangChain's LLM interface
    """
    
    # Instead of using pydantic properties, we'll use traditional Python attributes
    def __init__(self, model_path="gpt2"):
        """
        Initialize GPT-2 model
        
        Args:
            model_path (str): Path or HuggingFace model ID of the model
        """
    
    def __init__(self, model_path="gpt2"):
        """
        Initialize GPT-2 model
        
        Args:
            model_path (str): Path or HuggingFace model ID of the model
        """
        # Skip pydantic initialization to avoid validation errors
        # super().__init__() - Don't call this
        
        self.model_path = model_path
        self.model_name = "GPT-2"
        
        logger.info(f"Initializing GPT-2 model from {model_path}")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Set special tokens
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Successfully loaded GPT-2 model")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-2 model: {str(e)}")
            raise
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text based on input prompt
        
        Args:
            prompt (str): The input prompt for text generation
            max_length (int): Maximum length of generated text
            temperature (float): Temperature for sampling (higher = more random)
            
        Returns:
            str: Generated text response
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            output = self.model.generate(
                inputs["input_ids"],
                max_length=max_length + inputs["input_ids"].shape[1],
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and return only the generated part (excluding the prompt)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error in GPT-2 generation: {str(e)}")
            return f"Error in text generation: {str(e)}"
            
    def __call__(self, prompt, **kwargs):
        """
        Make the class callable for LangChain compatibility
        """
        return self.generate(prompt)
            
    def get_embedding(self, text):
        """
        Get embedding for text
        
        Args:
            text (str): Input text to embed
            
        Returns:
            list: Embedding vector
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # Use the last hidden state of the last layer as embedding
            last_hidden_state = outputs.hidden_states[-1]
            embedding = last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return None
