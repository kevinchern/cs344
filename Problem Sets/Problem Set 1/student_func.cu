// Homework 1
// Color to Greyscale Conversion

// A common way to represent color images is known as RGBA - the color
// is specified by how much Red, Grean and Blue is in it.
// The 'A' stands for Alpha and is used for transparency, it will be
// ignored in this homework.

// Each channel Red, Blue, Green and Alpha is represented by one byte.
// Since we are using one byte for each color there are 256 different
// possible values for each color.  This means we use 4 bytes per pixel.

// Greyscale images are represented by a single intensity value per pixel
// which is one byte in size.

// To convert an image from color to grayscale one simple method is to
// set the intensity to the average of the RGB channels.  But we will
// use a more sophisticated method that takes into account how the eye
// perceives color and weights the channels unequally.

// The eye responds most strongly to green followed by red and then blue.
// The NTSC (National Television System Committee) recommends the following
// formula for color to greyscale conversion:

// I = .299f * R + .587f * G + .114f * B

// Notice the trailing f's on the numbers which indicate that they are
// single precision floating point constants and not double precision
// constants.

// You should fill in the kernel as well as set the block and grid sizes
// so that the entire image is processed.

#include "utils.h"

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage,
                                  int numRows, int numCols)
{
  // TODO
  // Fill in the kernel to convert from color to greyscale
  // the mapping from components of a uchar4 to RGBA is:
  //  .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  // The output (greyImage) at each pixel should be the result of
  // applying the formula: output = .299f * R + .587f * G + .114f * B;
  // Note: We will be ignoring the alpha channel for this conversion

  // First create a mapping from the 2D block and grid locations
  // to an absolute 2D location in the image, then use that to
  // calculate a 1D offset

  // ====== KC below ======
  // the kernel launches many blocks and threads.

  // the first thing we need to figure out is, which thread are we on?
  // threadIdx.x, .y, and blockDim.x, .y will give us the info we need to determine "where" this
  // thread is.
  // once we know where the thread is, we need to assign a single pixel to it to process.
  // Essentially all we are doing here is defining a map from pixel index to thread index.

  // "global positioning" = intra-block offset + inter-block offset
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  // pixel index
  int pixel_idx = x + y * numCols;
  // I think the rgb image will reside in global memory, that's why we can access the single pixel
  // with rgbaImage[thIdx];
  uchar4 rgba = rgbaImage[pixel_idx];
  greyImage[pixel_idx] = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
}

void your_rgba_to_greyscale(const uchar4 *const h_rgbaImage, uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage, size_t numRows, size_t numCols)
{

  int const nThreadsX = 32;
  int const nThreadsY = nThreadsX;
  // how should i determine blocksize in general and in practice?
  const dim3 blockSize(nThreadsX, nThreadsY, 1);
  // one more than necessary
  unsigned int nBlocksY = (unsigned int)numRows / nThreadsX + 1;
  unsigned int nBlocksX = (unsigned int)numCols / nThreadsY + 1;
  const dim3 gridSize(nBlocksX, nBlocksY, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
