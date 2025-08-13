#!/usr/bin/env python3
"""
Convert learned_reducer.pt to two_tower_reducer.pt format
"""

import torch
import os

def convert_reducer_format():
    """Convert learned_reducer.pt to two_tower_reducer format"""
    
    # Load the new model
    if not os.path.exists('learned_reducer.pt'):
        print("❌ learned_reducer.pt not found")
        return False
    
    learned_model = torch.load('learned_reducer.pt', map_location='cpu')
    print("✅ Loaded learned_reducer.pt")
    print("Keys in learned model:", list(learned_model.keys()))
    
    # Convert to two_tower format
    two_tower_format = {
        'tower_state': learned_model.get('model_state', learned_model),
        'val_acc': learned_model.get('val_acc', 1.0),
        'output_dim': learned_model.get('output_dim', 8),
        'input_dim': 1536,
        'training_info': {
            'epochs': 50,
            'final_loss': learned_model.get('val_loss', 0.0)
        }
    }
    
    # Save in compatible format
    torch.save(two_tower_format, 'two_tower_reducer_v2.pt')
    print("✅ Saved as two_tower_reducer_v2.pt")
    
    # Backup original and replace
    if os.path.exists('two_tower_reducer.pt'):
        torch.save(torch.load('two_tower_reducer.pt'), 'two_tower_reducer_backup.pt')
        print("✅ Backed up original two_tower_reducer.pt")
    
    torch.save(two_tower_format, 'two_tower_reducer.pt')
    print("✅ Replaced two_tower_reducer.pt with new model")
    
    return True

if __name__ == "__main__":
    success = convert_reducer_format()
    if success:
        print("\n🎉 Conversion complete! The new reducer is now ready.")
    else:
        print("\n❌ Conversion failed.")