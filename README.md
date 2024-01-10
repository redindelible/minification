To compress a folder, use `cargo run -r -- compress <folder> <output file name>`. The output file name is optional.
To extract a folder, use `cargo run -r -- extract <compressed file> <output folder>`. Once again, the output folder is optional.

For reference, I used the Bad Apple! images available at [this](https://archive.org/details/bad_apple_is.7z) website 
as the compression test. Indeed, this algorithm was designed specifically to compress the frames of Bad Apple! as much as
possible, because it sounded like fun. The uncompressed folder of input images (converted to black-and-white, and with 
the visual artifacts at the edges of the images removed) is approximately 100 MB. 7zip can compress that to 28 MB. This 
algorithm can get that down to only 6.5 MB.

I used chain-coding (essentially, looking at the shapes in the images and recording their boundaries), as it would
be particularly well suited to Bad Apple!, as it is essentially all just a set of black and white blobs, which chain-coding
best at compressing. 

The biggest limitation of my current approach is that it can only compress black-and-white images, although this isn't
a fundamental limitation of the algorithm I coded in `chain_coding`. It's simply a means to get a working prototype out
faster.
