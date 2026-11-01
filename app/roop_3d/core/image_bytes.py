"""Inspect PNG/JPEG dimensions before allowing native image expansion."""
import cv2
import numpy as np
from roop_3d.core.errors import DomainError


def decode_face_image(data: bytes, max_pixels: int = 16_000_000) -> np.ndarray:
    try:
        if len(data)>16_500_000: raise ValueError()
        if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data)>=33 and data[12:16]==b'IHDR':
            width,height=int.from_bytes(data[16:20],'big'),int.from_bytes(data[20:24],'big')
        elif data.startswith(b'\xff\xd8'):
            cursor=2; width=height=0
            while cursor<len(data):
                if data[cursor]!=255: raise ValueError()
                while cursor<len(data) and data[cursor]==255: cursor+=1
                if cursor>=len(data): raise ValueError()
                marker=data[cursor];cursor+=1
                if marker in (0xD8,0x01) or 0xD0<=marker<=0xD7: continue
                if marker in (0xDA,0xD9): break
                if cursor+2>len(data): raise ValueError()
                size=int.from_bytes(data[cursor:cursor+2],'big')
                if size<2 or cursor+size>len(data): raise ValueError()
                if marker in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
                    if size<8: raise ValueError()
                    height=int.from_bytes(data[cursor+3:cursor+5],'big')
                    width=int.from_bytes(data[cursor+5:cursor+7],'big')
                    break
                cursor+=size
        else: raise ValueError()
        if width<=0 or height<=0 or width*height>max_pixels: raise ValueError()
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[0]*image.shape[1]>max_pixels: raise ValueError()
        return image
    except (ValueError,cv2.error):
        raise DomainError('IMAGE_DECODE_FAILED','Expected a valid PNG or JPEG within the 16 megapixel image budget',422) from None
