try:
    from .tabular_data_loader import load_train_data, load_val_data
except Exception:
    try:
        from .image_data_loader import load_train_data, load_val_data
    except Exception:
        pass