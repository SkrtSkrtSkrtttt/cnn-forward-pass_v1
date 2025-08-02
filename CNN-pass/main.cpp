//Naafiul Hossain
//SBU ID: 115107623
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include <cassert>
#include <complex>
using namespace std;

class ConvNet
{
private:
	// Stage 1 Input
	// Layer 1; Input layer
	vector<vector<vector<float>>> input;					// D1: Member to store 32x32x3 array of floating num of read input image

	// Stage 2: First Convolutional Layer + ReLU + Maxpooling
	// Layer 2: Convolution1, Stride = 1
	vector<vector<vector<float>>> conv_1;					// D2: Member to store 32x32x16 array of floating num of output of first convolution layer
	vector<vector<vector<vector<float>>>> filterOne;			// D3: Member to store 5x5x3x16 array of 16 convolution filters with zero padding
	vector<float> bias1;									// D4: Member to store 16 floating num of bias for 16 filters
	vector<vector<vector<float>>> FirstRelu;                    // D5: Member of 32x32x16 array to store output of first ReLU activation function
	vector<vector<vector<float>>> maxpool1;					// D6: Member of 16x16x16 array to store output of first maxpooling layer	

	// Stage 3: Second Convolutional Layer + ReLU + Maxpooling
	// Layer 5: Convolution2, Stride = 1
	vector<vector<vector<float>>> conv_2;					// D7: Member to store 16x16x20 array of floating num of output of second convolution layer
	vector<vector<vector<vector<float>>>> filterTwo;			// D8: Member to store 5x5x16x20 array of 20 convolution filters with zero padding
	vector<float> bias2;									// D9: Member to store 20 floating num of bias for 20 filters
	vector<vector<vector<float>>> TwoRelu;                    // D10: Member of 16x16x20 array to store output of second ReLU activation function
	vector<vector<vector<float>>> maxpool2;					// D11: Member of 8x8x20 array to store output of second maxpooling layer

	// Stage 4: Third Convolutional Layer + ReLU + Maxpooling
	// Layer 6: Convolution3, Stride = 1
	vector<vector<vector<float>>> conv_3;					// D12: Member to store 8x8x20 array of floating num of output of third convolution layer
	vector<vector<vector<vector<float>>>> filterThree;			// D13: Member to store 5x5x20x20 array of 20 convolution filters with zero padding
	vector<float> bias3;									// D14: Member to store 20 floating num of bias for 20 filters
	vector<vector<vector<float>>> ThreeRelu;                    // D15: Member of 8x8x20 array to store output of third ReLU activation function
	vector<vector<vector<float>>> maxpool3;					// D16: Member of 4x4x20 array to store output of third maxpooling layer

	// Stage 5: Fully Connected Layer + Softmax
	// Layer 11: Fully Connected Layer
	vector<float> LayerFull;                                // D17: Member to store 1x10 array of floating num of output of fully connected layer
	vector<vector<vector<vector<float>>>> fullFilter;		// D18: Member of 4x4x20x10 array for storing 10 full connection filters (dot product)
	vector<float> fullBias;									// D19: Member for array size of 10 for storing a bias vector

	// Layer 12: Softmax Layer
	vector<float> _SoftmaxLayer;								// D20: Member to store 1x10 array of floating num of output of softmax layer


public:
	ConvNet()
	{
		// Initialize input vector
		input.resize(32, vector<vector<float>>(32, vector<float>(3)));

		// Initialize conv1 vector
		conv_1.resize(32, vector<vector<float>>(32, vector<float>(16)));

		// Initialize filter1 vector
		filterOne.resize(16, vector<vector<vector<float>>>(3, vector<vector<float>>(5, vector<float>(5))));

		// Initialize bias1 vector
		bias1.resize(16);

		// Initialize relu1 vector
		FirstRelu.resize(32, vector<vector<float>>(32, vector<float>(16)));

		// Initialize maxpool1 vector
		maxpool1.resize(16, vector<vector<float>>(16, vector<float>(16)));

		// Initialize conv2 vector
		conv_2.resize(16, vector<vector<float>>(16, vector<float>(20)));

		// Initialize filter2 vector
		filterTwo.resize(20, vector<vector<vector<float>>>(16, vector<vector<float>>(5, vector<float>(5))));

		// Initialize bias2 vector
		bias2.resize(20);

		// Initialize relu2 vector
		TwoRelu.resize(16, vector<vector<float>>(16, vector<float>(20)));

		// Initialize maxpool2 vector
		maxpool2.resize(8, vector<vector<float>>(8, vector<float>(20)));

		// Initialize conv3 vector
		conv_3.resize(8, vector<vector<float>>(8, vector<float>(20)));

		// Initialize filter3 vector
		filterThree.resize(20, vector<vector<vector<float>>>(20, vector<vector<float>>(5, vector<float>(5))));

		// Initialize bias3 vector
		bias3.resize(20);

		// Initialize relu3 vector
		ThreeRelu.resize(8, vector<vector<float>>(8, vector<float>(20)));

		// Initialize maxpool3 vector
		maxpool3.resize(4, vector<vector<float>>(4, vector<float>(20)));

		// Initialize fullLayer vector
		LayerFull.resize(10);

		// Initialize fullFilter vector
		fullFilter.resize(10, vector<vector<vector<float>>>(20, vector<vector<float>>(4, vector<float>(4))));

		// Initialize fullBias vector
		fullBias.resize(10);

		// Initialize softmaxLayer vector
		_SoftmaxLayer.resize(10);
	}

	// Stage 1 Input
	// Layer 1; Input layer
	// M1: read input file and initialize input layer with input image
	void read_input_image(string input_image)
	{
		// Read input image from file
		ifstream file(input_image);
		cout << "After reading input image:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 32; j++)
				{
					for (int k = 0; k < 32; k++)
					{
						file >> input[j][k][i];
						cout << input[j][k][i] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
		}
		file.close();
	}


	// Stage 2: First Convolutional Layer + ReLU + Maxpooling
	// Layer 2: Convolution, Stride = 1
	// M2: read filter weights from CNN_weights.txt and initialize filter1 with filter weights
	void scan_filter_weights(string filename)
	{
		// Read filter weights from file
		ifstream file(filename);
		cout << "After reading filter weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							file >> filterOne[i][j][k][l];
							cout << filterOne[i][j][k][l] << " ";
						}
						cout << endl;
					}
					cout << endl;
					cout << endl;
				}
			}
		}
		file.close();
	}

	// M3: read bias weights from CNN_weights.txt and initialize bias1 with bias weights
	// bias weights are the 16 floating num after the 5x5x3x16 filter weights
	void scan_bias_weights(string filename)
	{
		// Read bias weights from file
		ifstream file(filename);
		cout << "After reading bias weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 16; i++)
			{
				file >> bias1[i];
				cout << bias1[i] << " ";
			}
			cout << endl;
			cout << endl;
		}
		file.close();
	}

	// M4: Method to perform convolution operation on input and filter1, adding bias1 and store in conv1 member
	void convolution1()
	{
		ofstream outfile("conv1_output.txt");
		int stride = 1;
		int inputHeight = input.size(); // Height of input
		int inputWidth = input[0].size(); // Width of input
		int inputChannels = input[0][0].size(); // Number of channels in input

		int convHeight = conv_1.size(); // Height of conv1
		int convWidth = conv_1[0].size(); // Width of conv1
		int numFilters = filterOne.size(); // Number of filters in filter1
		int filterHeight = filterOne[0].size(); // Height of each filter
		int filterWidth = filterOne[0][0].size(); // Width of each filter

		// Initialize conv1 to match the dimensions of input
		conv_1.resize(convHeight, vector<vector<float>>(convWidth, vector<float>(numFilters)));

		cout << "Output of Convolution layer: conv1" << endl;
		for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex)
		{
			for (int convRow = 0; convRow < convHeight; ++convRow)
			{
				for (int convCol = 0; convCol < convWidth; ++convCol)
				{
					float sum = 0.0f;

					for (int channel = 0; channel < inputChannels; ++channel)
					{
						for (int filterRow = 0; filterRow < filterHeight; ++filterRow)
						{
							for (int filterCol = 0; filterCol < filterWidth; ++filterCol)
							{
								int inputRow = convRow + filterRow - filterHeight / 2;
								int inputCol = convCol + filterCol - filterWidth / 2;

								if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
								{
									sum += input[inputRow][inputCol][channel] * filterOne[filterIndex][channel][filterRow][filterCol];
								}
							}
						}
					}

					conv_1[convRow][convCol][filterIndex] = sum + bias1[filterIndex];
					cout << conv_1[convRow][convCol][filterIndex] << "  ";
					outfile << conv_1[convRow][convCol][filterIndex] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}


	// Layer 3: ReLU activation function
	// M5: Method to compute relu1 using conv1
	// ReLU activation function: relu1 = max(0, conv1)
	void reluActivation1()
	{
		ofstream outfile("relu1_output.txt");
		int tn1s1 = conv_1.size(); // Dimension 1 of conv1
		int tn1s2 = conv_1[0].size(); // Dimension 2 of conv1
		int tn1s3 = conv_1[0][0].size(); // Dimension 3 of conv1

		// Initialize relu1 to match the dimensions of conv1
		FirstRelu.resize(tn1s1, vector<vector<float>>(tn1s2, vector<float>(tn1s3)));
		cout << "After ReLU activation function:" << endl;
		for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2++)
			{
				for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1++)
				{
					FirstRelu[tn1i1][tn1i2][tn1i3] = max(0.0f, conv_1[tn1i1][tn1i2][tn1i3]);
					cout << FirstRelu[tn1i1][tn1i2][tn1i3] << "  ";
					outfile << FirstRelu[tn1i1][tn1i2][tn1i3] << "  ";
				}

				cout << endl;
				outfile << endl;
			}

			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Layer 4: Maxpooling (filter size 2x2, stride 2)
	// M6: Method to process relu1 to compute and store maxpool1 
	// Each element is max of 4 elements in a 2x2 window, Stride 2
	void maxPooling1()
	{
		ofstream outfile("maxpool1_output.txt");
		int stride = 2;
		int tn1s1 = FirstRelu.size(); // Dimension 1 of relu1
		int tn1s2 = FirstRelu[0].size(); // Dimension 2 of relu1
		int tn1s3 = FirstRelu[0][0].size(); // Dimension 3 of relu1

		// Initialize maxpool1 to match the dimensions of relu1
		maxpool1.resize(tn1s1 / stride, vector<vector<float>>(tn1s2 / stride, vector<float>(tn1s3)));
		cout << "After maxpooling:" << endl;
		for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1 += stride)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2 += stride)
			{
				for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
				{
					float maxval = 0.0f;
					for (int tn1i1s = 0; tn1i1s < stride; tn1i1s++)
					{
						for (int tn1i2s = 0; tn1i2s < stride; tn1i2s++)
						{
							maxval = max(maxval, FirstRelu[tn1i1 + tn1i1s][tn1i2 + tn1i2s][tn1i3]);
						}
					}

					maxpool1[tn1i1 / stride][tn1i2 / stride][tn1i3] = maxval;
					cout << maxpool1[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
					outfile << maxpool1[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
				}

				cout << endl;
				outfile << endl;
			}

			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Stage 3: Second Convolutional Layer + ReLU + Maxpooling
	// Layer 5: Convolution, Stride = 1
	// M7: Method to read filter weights from CNN_weights.txt and initialize filter2 with filter weights
	void read_filter2_weights(string filename)
	{
		// Read filter weights from file
		ifstream file(filename);
		cout << "After reading filter2 weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							file >> filterTwo[i][j][k][l];
							cout << filterTwo[i][j][k][l] << " ";
						}
						cout << endl;
					}
					cout << endl;
					cout << endl;
				}
			}
		}
		file.close();
	}

	// M8: Method to read bias from input file and store in bias2 member for initialization
	void read_bias2_weights(string filename)
	{
		// Read bias weights from file
		ifstream file(filename);
		cout << "After reading bias2 weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 20; i++) // <-- Ensure this loop iterates 20 times for bias2
			{
				file >> bias2[i]; // <-- Access elements from bias2 vector
				cout << bias2[i] << " ";
			}
			cout << endl;
			cout << endl;
		}
		file.close();
	}

	// M9: Method to perform convolution operation on maxpool1 and filter2, adding bias2 and store in conv2 member, stride = 1
	void convolution2()
	{
		ofstream outfile("conv2_output.txt");
		int stride = 1;
		int tn1s1 = maxpool1.size(); // Dimension 1 of maxpool1
		int tn1s2 = maxpool1[0].size(); // Dimension 2 of maxpool1
		int tn1s3 = maxpool1[0][0].size(); // Dimension 3 of maxpool1
		int tn2s1 = filterTwo[0][0].size(); // Dimension 1 of filter2
		int tn2s2 = filterTwo[0][0][0].size(); // Dimension 2 of filter2
		int tn2s3 = filterTwo[0].size(); // Dimension 3 of filter2
		int tn2s4 = filterTwo.size(); // Dimension 4 of filter2
		int tn2s1by2 = tn2s1 / 2; // Half of Dimension 1 of filter2
		int tn2s2by2 = tn2s2 / 2; // Half of Dimension 2 of filter2

		// Initialize conv2 to match the dimensions of maxpool1
		conv_2.resize(tn1s1, vector<vector<float>>(tn1s2, vector<float>(tn2s4)));

		// Perform convolution operation
		cout << "Output of Convolution layer: conv2" << endl;
		for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++)
		{
			for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++)
			{
				for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++)
				{
					double tmpsum = 0.0;
					for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++)
					{
						// note tn1s3=tn2s3
						for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++)
						{
							for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++)
							{
								if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) < tn1s2)) // zero padding of tn1
								{
									tmpsum += filterTwo[tn2i4][tn2i3][tn2i1 + tn2s1by2][tn2i2 + tn2s2by2] * maxpool1[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
								}
							}
						}
					}
					conv_2[tn3i1][tn3i2][tn3i3] = tmpsum + bias2[tn3i3];
					cout << conv_2[tn3i1][tn3i2][tn3i3] << "  ";
					outfile << conv_2[tn3i1][tn3i2][tn3i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Layer 6: ReLU activation function
	// M10: Method to compute relu2 using conv2
	// ReLU activation function: relu2 = max(0, conv2)
	void reluActivation2()
	{
		ofstream outfile("relu2_output.txt");
		int tn1s1 = conv_2.size();         // Dimension 1 of conv2
		int tn1s2 = conv_2[0].size();      // Dimension 2 of conv2
		int tn1s3 = conv_2[0][0].size();   // Dimension 3 of conv2

		// Initialize relu2 to match the dimensions of conv2
		TwoRelu.resize(tn1s1, vector<vector<float>>(tn1s2, vector<float>(tn1s3)));
		cout << "After ReLU activation function:" << endl;
		for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2++)
			{
				for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1++)
				{
					TwoRelu[tn1i1][tn1i2][tn1i3] = max(0.0f, conv_2[tn1i1][tn1i2][tn1i3]);
					cout << TwoRelu[tn1i1][tn1i2][tn1i3] << "  ";
					outfile << TwoRelu[tn1i1][tn1i2][tn1i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Layer 7: Maxpooling (filter size 2x2, stride 2)
	// M11: Method to process relu2 to compute and store maxpool2
	// Each element is max of 4 elements in a 2x2 window, Stride 2
	void maxPooling2()
	{
		ofstream outfile("maxpool2_output.txt");
		int stride = 2;
		int tn1s1 = TwoRelu.size(); // Dimension 1 of relu2
		int tn1s2 = TwoRelu[0].size(); // Dimension 2 of relu2
		int tn1s3 = TwoRelu[0][0].size(); // Dimension 3 of relu2

		// Initialize maxpool2 to match the dimensions of relu2
		maxpool2.resize(tn1s1 / stride, vector<vector<float>>(tn1s2 / stride, vector<float>(tn1s3)));

		cout << "After maxpooling:" << endl;
		for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2 += stride)
			{
				for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1 += stride)
				{
					float maxval = 0.0f;
					for (int tn1i2s = 0; tn1i2s < stride; tn1i2s++)
					{
						for (int tn1i1s = 0; tn1i1s < stride; tn1i1s++)
						{
							maxval = max(maxval, TwoRelu[tn1i1 + tn1i1s][tn1i2 + tn1i2s][tn1i3]);
						}
					}
					maxpool2[tn1i1 / stride][tn1i2 / stride][tn1i3] = maxval;
					cout << maxpool2[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
					outfile << maxpool2[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Stage 4: Third Convolutional Layer + ReLU + Maxpooling
	// Layer 8: Convolution3, Stride = 1
	// M12: Method to read filter weights from CNN_weights.txt and initialize filter3 with filter weights
	void read_filter3_weights(string filename)
	{
		// Read filter weights from file
		ifstream file(filename);
		cout << "After reading filter3 weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 20; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							file >> filterThree[i][j][k][l];
							cout << filterThree[i][j][k][l] << " ";
						}
						cout << endl;
					}
					cout << endl;
					cout << endl;
				}
			}
		}
	}

	// M13: Method to read bias from input file and store in bias3 member for initialization
	void read_bias3_weights(string filename)
	{
		// Read bias weights from file
		ifstream file(filename);
		cout << "After reading bias3 weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 20; i++) // <-- Ensure this loop iterates 20 times for bias3
			{
				file >> bias3[i]; // <-- Access elements from bias3 vector
				cout << bias3[i] << " ";
			}
			cout << endl;
			cout << endl;
		}
		file.close();
	}

	// M14 : Method to perform convolution operation on maxpool2 and filter3, adding bias3 and store in conv3 member, stride = 1
	void convolution3()
	{
		ofstream outfile("conv3_output.txt");
		int stride = 1;
		int tn1s1 = maxpool2.size(); // Dimension 1 of maxpool2
		int tn1s2 = maxpool2[0].size(); // Dimension 2 of maxpool2
		int tn1s3 = maxpool2[0][0].size(); // Dimension 3 of maxpool2
		int tn2s1 = filterThree[0][0].size(); // Dimension 1 of filter3
		int tn2s2 = filterThree[0][0][0].size(); // Dimension 2 of filter3
		int tn2s3 = filterThree[0].size(); // Dimension 3 of filter3
		int tn2s4 = filterThree.size(); // Dimension 4 of filter3
		int tn2s1by2 = tn2s1 / 2; // Half of Dimension 1 of filter3
		int tn2s2by2 = tn2s2 / 2; // Half of Dimension 2 of filter3

		// Initialize conv3 to match the dimensions of maxpool2
		conv_3.resize(tn1s1, vector<vector<float>>(tn1s2, vector<float>(tn2s4)));

		// Perform convolution operation
		cout << "Output of Convolution layer: conv3" << endl;
		for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++)
		{
			for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++)
			{
				for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++)
				{
					double tmpsum = 0.0;
					for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++)
					{
						// note tn1s3=tn2s3
						for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++)
						{
							for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++)
							{
								if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) < tn1s2)) // zero padding of tn1
								{
									tmpsum += filterThree[tn2i4][tn2i3][tn2i1 + tn2s1by2][tn2i2 + tn2s2by2] * maxpool2[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
								}
							}
						}
					}

					conv_3[tn3i1][tn3i2][tn3i3] = tmpsum + bias3[tn3i3];
					cout << conv_3[tn3i1][tn3i2][tn3i3] << "  ";
					outfile << conv_3[tn3i1][tn3i2][tn3i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Layer 9: ReLU activation function
	// M15: Method to compute relu3 using conv3
	// ReLU activation function: relu3 = max(0, conv3)
	void ThirdReluActive()
	{
		ofstream outfile("relu3_output.txt");
		int tn1s1 = conv_3.size();         // Dimension 1 of conv3
		int tn1s2 = conv_3[0].size();      // Dimension 2 of conv3
		int tn1s3 = conv_3[0][0].size();   // Dimension 3 of conv3

		// Initialize relu3 to match the dimensions of conv3
		ThreeRelu.resize(tn1s1, vector<vector<float>>(tn1s2, vector<float>(tn1s3)));
		cout << "After ReLU activation function:" << endl;
		for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2++)
			{
				for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1++)
				{
					ThreeRelu[tn1i1][tn1i2][tn1i3] = max(0.0f, conv_3[tn1i1][tn1i2][tn1i3]);
					cout << ThreeRelu[tn1i1][tn1i2][tn1i3] << "  ";
					outfile << ThreeRelu[tn1i1][tn1i2][tn1i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Layer 10: Maxpooling (filter size 2x2, stride 2)
	// M16: Method to process relu3 to compute and store maxpool3
	// Each element is max of 4 elements in a 2x2 window, Stride 2
	void maxPooling3()
	{
		ofstream outfile("maxpool3_output.txt");
		int stride = 2;
		int tn1s1 = ThreeRelu.size(); // Dimension 1 of relu3
		int tn1s2 = ThreeRelu[0].size(); // Dimension 2 of relu3
		int tn1s3 = ThreeRelu[0][0].size(); // Dimension 3 of relu3

		// Initialize maxpool3 to match the dimensions of relu3
		maxpool3.resize(tn1s1 / stride, vector<vector<float>>(tn1s2 / stride, vector<float>(tn1s3)));

		cout << "After maxpooling:" << endl;
		for (int tn1i3 = 0; tn1i3 < tn1s3; tn1i3++)
		{
			for (int tn1i2 = 0; tn1i2 < tn1s2; tn1i2 += stride)
			{
				for (int tn1i1 = 0; tn1i1 < tn1s1; tn1i1 += stride)
				{
					float maxval = 0.0f;
					for (int tn1i2s = 0; tn1i2s < stride; tn1i2s++)
					{
						for (int tn1i1s = 0; tn1i1s < stride; tn1i1s++)
						{
							maxval = max(maxval, ThreeRelu[tn1i1 + tn1i1s][tn1i2 + tn1i2s][tn1i3]);
						}
					}
					maxpool3[tn1i1 / stride][tn1i2 / stride][tn1i3] = maxval;
					cout << maxpool3[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
					outfile << maxpool3[tn1i1 / stride][tn1i2 / stride][tn1i3] << "  ";
				}
				cout << endl;
				outfile << endl;
			}
			cout << endl << endl;
			outfile << endl << endl;
		}
	}

	// Stage 5: Fully Connected Layer + Softmax
	// Layer 11: Fully Connected Layer
	// M17: Method to read full connection filter weights from CNN_weights.txt and initialize fullFilter with filter weights
	void read_fullFilter_weights(string filename)
	{
		// Read filter weights from file
		ifstream file(filename);
		cout << "After reading fullFilter weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 20; j++)
				{
					for (int k = 0; k < 4; k++)
					{
						for (int l = 0; l < 4; l++)
						{
							file >> fullFilter[i][j][k][l];
							cout << fullFilter[i][j][k][l] << " ";
						}
						cout << endl;
					}
					cout << endl;
					cout << endl;
				}
			}
		}
		file.close();
	}

	// M18: Method to read bias from input file and store in fullBias member for initialization
	void read_fullBias_weights(string filename)
	{
		// Read bias weights from file
		ifstream file(filename);
		cout << "After reading fullBias weights:" << endl;
		if (file.is_open())
		{
			for (int i = 0; i < 10; i++) // <-- Ensure this loop iterates 10 times for fullBias
			{
				file >> fullBias[i]; // <-- Access elements from fullBias vector
				cout << fullBias[i] << " ";
			}
			cout << endl;
			cout << endl;
		}
		file.close();
	}

	// M19: Method to compute fullLayer by taking dot product of maxpool3 and fullFilter, and adding fullBias
	void fullLayerCalculation()
	{
		int tn4s1 = fullFilter.size();         // Dimension 1 of fullFilter = 10
		int tn4s2 = fullFilter[0].size();      // Dimension 2 of fullFilter = 20
		int tn4s3 = fullFilter[0][0].size();   // Dimension 3 of fullFilter = 4
		int tn4s4 = fullFilter[0][0][0].size(); // Dimension 4 of fullFilter = 4

		cout << tn4s1 << " " << tn4s2 << " " << tn4s3 << " " << tn4s4 << endl;

		cout << "Output of Fully Connected Layer: fullLayer" << endl;
		for (int tn4i1 = 0; tn4i1 < 10; tn4i1++) // Corrected loop index to iterate over the fourth dimension of fullLayer
		{
			double tmpsum = 0.0;
			for (int tn4i2 = 0; tn4i2 < 20; tn4i2++)  // Iterate over the first dimension of fullFilter
			{
				for (int tn4i3 = 0; tn4i3 < 4; tn4i3++)  // Iterate over the second dimension of fullFilter
				{
					for (int tn4i4 = 0; tn4i4 < 4; tn4i4++)  // Iterate over the third dimension of fullFilter
					{
						//tmpsum += fullFilter[tn4i1][tn4i2][tn4i3][tn4i4] * maxpool3[tn4i1][tn4i2][tn4i3];
						tmpsum += fullFilter[tn4i1][tn4i2][tn4i3][tn4i4] * maxpool3[tn4i4][tn4i3][tn4i2];
						// 4 x 4 x 20 x 10 || 4 x 4 x 20 
						// 10 x 20 x 4 x 4 || 4 x 4 x 20
					}
				}
			}
			LayerFull[tn4i1] = tmpsum + fullBias[tn4i1];
			cout << LayerFull[tn4i1] << "  ";
		}
		cout << endl;
		cout << endl;
	}

	// Layer 12: Softmax Layer
	// M20: Method to normalize fullLayer to compute softmaxLayer
	// Divide each element by the square root of the sum of squares of all elements
	void softmaxLayerCalculation()
	{
		double sum_of_squares = 0.0;
		_SoftmaxLayer.resize(10);
		// Calculate the sum of squares of all elements
		for (int i = 0; i < 10; i++)
		{
			sum_of_squares += LayerFull[i] * LayerFull[i];
		}

		// Normalize each element by dividing by the square root of the sum of squares
		double normalization_factor = sqrt(sum_of_squares);
		cout << "Normalization Factor: " << normalization_factor << endl;
		cout << "Output of Softmax Layer: softmaxLayer" << endl;
		for (int i = 0; i < 10; i++)
		{
			_SoftmaxLayer[i] = LayerFull[i] / normalization_factor;
			cout << _SoftmaxLayer[i] << " = " << LayerFull[i] << " / " << normalization_factor << endl;
		}
		cout << endl << endl;
	}


	// M21: Method for softmax function to compute each output probability (exp(x)/(sum of exp(xi) for all i))
	void softmaxLayerProbabilityCalculation()
	{
		double sum_of_exps = 0.0;
		_SoftmaxLayer.resize(10);
		// Calculate the sum of exponentials of all elements
		for (int i = 0; i < 10; i++)
		{
			sum_of_exps += exp(_SoftmaxLayer[i]);
		}

		// Compute softmax function for each element
		cout << "Output of Softmax Layer Probability Calculation: " << endl;
		for (int i = 0; i < 10; i++)
		{
			_SoftmaxLayer[i] = exp(_SoftmaxLayer[i]) / sum_of_exps;
			cout << _SoftmaxLayer[i] << "  ";
		}
		cout << endl << endl;
	}

	// M22: Method to display the output probabilities
	void displayOutputProbabilities()
	{
		double testProbSum = 0.0;
		ofstream outfile("output_probabilities.txt");
		cout << "Output Probabilities: " << endl;
		for (int i = 0; i < 10; i++)
		{
			cout << "Probability of " << i << " is " << _SoftmaxLayer[i] << endl;
			outfile << "Probability of " << i << " is " << _SoftmaxLayer[i] << endl;
			testProbSum += _SoftmaxLayer[i];
		}
		// is the sum of all probabilities equal to 1?
		cout << "Sum of all probabilities: " << testProbSum << endl;
		outfile << "Sum of all probabilities: " << testProbSum << endl;
		cout << endl;
		outfile << endl;
	}

};

// generate random data for a 1d vector
void data1d(int s1, string filename) {
	srand((unsigned int)time(NULL));// seed rand() with system time
	//Generating input for project 1
	ofstream outfile(filename);
	for (int i1 = 0; i1 < s1; i1++) {
		outfile << (double)((rand() % 100) / 100.0) << "   ";
	}
	outfile << endl << endl;
}

// generate random data for a 2d tensor
void data2d(int s1, int s2, string filename) {
	srand((unsigned int)time(NULL));// seed rand() with system time
	//Generating input for project 1
	ofstream outfile(filename);
	for (int i2 = 0; i2 < s2; i2++) {
		for (int i1 = 0; i1 < s1; i1++) {
			outfile << (double)((rand() % 100) / 100.0) << "   ";
		}
		outfile << endl;
	}
	outfile << endl << endl;
}

// generate random data for a 3d tensor
void data3d(int s1, int s2, int s3, string filename) {
	srand((unsigned int)time(NULL));// seed rand() with system time
	//Generating input for project 1
	ofstream outfile(filename);
	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++) {
				outfile << (double)((rand() % 100) / 100.0) << "   ";
			}
			outfile << endl;
		}
		outfile << endl << endl;
	}
	outfile << endl << endl;
}

// generate random data for a 4d tensor
void data4d(int s1, int s2, int s3, int s4, string filename) {
	srand((unsigned int)time(NULL));// seed rand() with system time
	//Generating input for project 1
	ofstream outfile(filename);
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					outfile << (double)((rand() % 100) / 100.0) << "   ";
				}
				outfile << endl;
			}
			outfile << endl << endl;
		}
		outfile << endl << endl;
	}
	outfile << endl << endl;
}

int main(int argc, char* argv[])
{
	// Generate random input data for the CNN
	//data1d(16, "bias1.txt");
	//data4d(5, 5, 16, 20, "filter2.txt");
	//data1d(20, "bias2.txt");
	//data4d(5, 5, 20, 20, "filter3.txt");
	//data1d(20, "bias3.txt");
	//data4d(4, 4, 20, 10, "fullFilter.txt");
	//data1d(10, "fullBias.txt");
	ConvNet convnet;
	char c;

	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.read_input_image("Test_image.txt");
	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.scan_filter_weights("CNN_weights.txt");
	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.scan_bias_weights("bias1.txt");
	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.convolution1();
	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.reluActivation1();
	cout << "Enter any character to proceed: ";
	cin >> c;
	convnet.maxPooling1();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_filter2_weights("filter2.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_bias2_weights("bias2.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.convolution2();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.reluActivation2();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.maxPooling2();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_filter3_weights("filter3.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_bias3_weights("bias3.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.convolution3();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.ThirdReluActive();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.maxPooling3();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_fullFilter_weights("fullFiler.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.read_fullBias_weights("fullBias.txt");
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.fullLayerCalculation();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.softmaxLayerCalculation();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.softmaxLayerProbabilityCalculation();
	cout << "Enter any character to continue: ";
	cin >> c;
	convnet.displayOutputProbabilities();

	return 0;
}