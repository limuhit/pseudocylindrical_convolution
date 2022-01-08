# pseudocylindrical_convolution
Pseudocylindrical convolutions for Learned Omnidirectional Image Compression

Requirment packages:
- pytorch
- cv2 (python-opencv) 
 
Install:
* python setup.py install
* cd coder & python setup_linux.py install
	
Running the codec for 360-degree images:
* Encoding:
 	* python pseudo_codec.py --enc --img-file image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python pseudo_codec.py --enc --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
* Decoding:
 	* python pseudo_codec.py --dec --out-file decoded_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python pseudo_codec.py --dec --out-list a_dec.png b_dec.png --code-list code_a code_b --model-idx 3 --ssim
* Testing (Decoding and evaluate the performance):
 	* python pseudo_codec.py --test --img-file source_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
<<<<<<< HEAD
 	* python pseudo_codec.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
=======
 	* python pseudo_codec.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
		
>>>>>>> d1cfc16506d35af1654d1b4a70eef87522e72bab
