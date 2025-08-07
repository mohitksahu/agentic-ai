import os
import logging
import torch
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List

logger = logging.getLogger(__name__)

class DistilBertWrapper:
    """
    Wrapper class for DistilBERT model that implements LangChain's LLM interface
    """
    
    def __init__(self, model_path="distilbert-base-uncased"):
        """
        Initialize DistilBERT model
        
        Args:
            model_path (str): Path or HuggingFace model ID of the model
        """
        # Skip pydantic initialization to avoid validation errors
        # super().__init__() - Don't call this
        
        self.model_path = model_path
        self.model_name = "DistilBERT"
        
        logger.info(f"Initializing DistilBERT model from {model_path}")
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertModel.from_pretrained(model_path)
            
            # For text generation we need to use a different approach since DistilBERT isn't designed for generation
            # We'll use a language model-based pipeline instead for generating text
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            
            # Using GPT-2 for text generation
            generator_model = AutoModelForCausalLM.from_pretrained("gpt2")
            generator_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.generator = pipeline('text-generation', model=generator_model, tokenizer=generator_tokenizer)
            
            logger.info(f"Successfully loaded DistilBERT model")
        except Exception as e:
            logger.error(f"Failed to initialize DistilBERT model: {str(e)}")
            raise
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text based on input prompt
        Note: DistilBERT is not a generative model, so we use GPT-2 for this functionality
        
        Args:
            prompt (str): The input prompt for text generation
            max_length (int): Maximum length of generated text
            temperature (float): Temperature for sampling (higher = more random)
            
        Returns:
            str: Generated text response
        """
        try:
            # Using the generator pipeline
            outputs = self.generator(
                prompt, 
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            
            # Extract the generated text, removing the prompt
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error in DistilBERT generation: {str(e)}")
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
                outputs = self.model(**inputs)
                
            # Use the last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return None
