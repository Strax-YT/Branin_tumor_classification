
                best_accuracy = accuracy
                best_params = params
                classifier.save_model(f'best_hyperparams_model.pth')
                
        except Exception as e:
            print(f"Error with parameter set {i+1}: {e}")
            continue
    
    # Summary
    print(f"\nHYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 50)