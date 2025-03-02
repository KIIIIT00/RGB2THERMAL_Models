import timm
from timm.models.swin_transformer import SwinTransformerBlock

swin_models = [model for model in timm.list_models() if 'swin' in model]
print(swin_models)
block = timm.create_model('swin_s3_base_224', pretrained=False, num_classes=0, drop_rate=0.0)
print("Model:", block.layers[0].blocks[0])
# block_224 = timm.create_model("swin_s3_base_224", pretrained=False, num_classes = 0, drop_rate=0.0)
# print("Block_224:", block_224)