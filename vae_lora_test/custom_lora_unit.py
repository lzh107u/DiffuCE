import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict

class LoRA_Conv2d( nn.Module ):
    def __init__( 
        self, 
        in_channels     : int, 
        out_channels    : int, 
        kernel_size     : int = 3, 
        stride          : int = 1, 
        padding         : int = 1,
        latent_factor   : int = 4, ) -> None:
        super().__init__()
        
        self.fixed_unit = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding )
        self.lora_a = nn.Conv2d( in_channels = in_channels, out_channels = 1, kernel_size = 3, stride = latent_factor, padding = 1 )
        self.lora_b = nn.Conv2d( in_channels = 1, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1 )

        self.lora_init()

    def lora_init( self ):
        print( "conv2d init called" )
        init_funcs : List[ Callable[ [ torch.Tensor ], torch.Tensor ] ] = [   
            torch.randn_like, # Gaussian init for A
            torch.zeros_like # zero init for B
        ]
        layers : List[ nn.Module ] = [
            self.lora_a,
            self.lora_b
        ]

        for cnt, layer in enumerate( layers ):
            state_dict : Dict[ str, torch.Tensor ] = layer.state_dict()
            func = init_funcs[ cnt ]
            for name in state_dict:
                weight = func( state_dict[ name ] )
                state_dict[ name ] = weight
            layer.load_state_dict( state_dict = state_dict )

    def lora_train( self ):
        self.fixed_unit.eval()
        self.lora_a.train()
        self.lora_b.train()

    def forward( self, x : torch.Tensor ) -> torch.Tensor:
        # 模型通過固定的區域
        self.fixed_unit.eval()
        with torch.no_grad():
            fixed_output : torch.Tensor = self.fixed_unit( x )

        # 模型通過學習區
        # 這裡在 lora_a 縮小 feature map 的長寬
        # 在 lora_b 之後需要將 feature map 的長寬放大回去
        out : torch.Tensor = self.lora_a( x )
        out = self.lora_b( out )
        out = F.interpolate( input = out, size = fixed_output.shape[ -2 : ], mode = 'nearest' )  
        out = out + fixed_output
        return out

class LoRA_linear( nn.Module ):
    def __init__( 
        self,
        in_features     : int,
        out_features    : int,
        latent_deg      : int = 4 ):
        super().__init__()

        self.fixed_unit = nn.Linear( in_features = in_features, out_features = out_features )
        self.lora_a = nn.Linear( in_features = in_features, out_features = latent_deg )
        self.lora_b = nn.Linear( in_features = latent_deg, out_features = out_features )

        self.lora_init()

    def lora_init( self ):
        print( "linear init called" )
        init_funcs : List[ Callable[ [ torch.Tensor ], torch.Tensor ] ] = [   
            torch.randn_like, # Gaussian init for A
            torch.zeros_like # zero init for B
        ]
        layers : List[ nn.Module ] = [
            self.lora_a,
            self.lora_b
        ]

        for cnt, layer in enumerate( layers ):
            state_dict : Dict[ str, torch.Tensor ] = layer.state_dict()
            func = init_funcs[ cnt ]
            for name in state_dict:
                weight = func( state_dict[ name ] )
                state_dict[ name ] = weight
            layer.load_state_dict( state_dict = state_dict )

    def forward( self, x : torch.Tensor ) -> torch.Tensor:
        # 模型通過固定的區域
        self.fixed_unit.eval()
        with torch.no_grad():
            fixed_output : torch.Tensor = self.fixed_unit( x )

        # 模型通過學習區
        out : torch.Tensor = self.lora_a( x )
        out = self.lora_b( out )

        out = out + fixed_output
            
        return out