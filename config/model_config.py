from dataclasses import dataclass

@dataclass
class ModelConfig:
    h: int = 1  # forecasting horizon
    look_back_window: int = 24  # 24 hours
    input_dim: int = 3
    output_dim: int = 1
    num_epochs: int = 2
    lr: float = 0.01
    batch_size: int = 32
    
    def __post_init__(self):
        # This method is automatically called after the class is initialized.
        # It's used to set additional attributes or perform post-initialization logic.
        self.model_name = f'GSPHAR_24_magnet_dynamic_h{self.h}'