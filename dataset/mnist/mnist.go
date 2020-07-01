// http://yann.lecun.com/exdb/mnist/

package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"github.com/minami14/tengor/nn"
	"io"
	"net/http"
	"os"
)

const (
	url        = "http://yann.lecun.com/exdb/mnist/"
	trainImage = "train-images-idx3-ubyte.gz"
	trainLabel = "train-labels-idx1-ubyte.gz"
	testImage  = "t10k-images-idx3-ubyte.gz"
	testLabel  = "t10k-labels-idx1-ubyte.gz"

	imageMagic = 2051
	labelMagic = 2049

	baseDir        = ".tengor/dataset/mnist/"
	trainImagePath = baseDir + trainImage
	trainLabelPath = baseDir + trainLabel
	testImagePath  = baseDir + testImage
	testLabelPath  = baseDir + testLabel
)

func LoadImage(path string) ([]*nn.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	reader, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()

	buf := make([]byte, 28*28)
	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	n := binary.BigEndian.Uint32(buf[:4])
	if n != imageMagic {
		return nil, fmt.Errorf("invalid magic number: %v", n)
	}

	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	items := int(binary.BigEndian.Uint32(buf[:4]))

	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	h := int(binary.BigEndian.Uint32(buf[:4]))

	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	w := int(binary.BigEndian.Uint32(buf[:4]))

	size := h * w

	images := make([]*nn.Tensor, items)
	for i := 0; i < items; i++ {
		if _, err := reader.Read(buf[:size]); err != nil && err != io.EOF {
			return nil, err
		}

		data := make([]float64, size)
		for j := 0; j < size; j++ {
			data[j] = float64(buf[j]) / 255
		}
		images[i] = nn.TensorFromSlice(nn.Shape{h, w}, data)
	}

	return images, nil
}

func LoadLabel(path string) ([]*nn.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	reader, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()

	buf := make([]byte, 28*28)
	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	n := binary.BigEndian.Uint32(buf[:4])
	if n != labelMagic {
		return nil, fmt.Errorf("invalid magic number: %v", n)
	}

	if _, err := reader.Read(buf[:4]); err != nil {
		return nil, err
	}

	items := int(binary.BigEndian.Uint32(buf[:4]))

	labels := make([]*nn.Tensor, items)
	for i := 0; i < items; i++ {
		if _, err := reader.Read(buf[:1]); err != nil && err != io.EOF {
			return nil, err
		}

		data := make([]float64, 10)
		data[buf[0]] = 1
		labels[i] = nn.TensorFromSlice(nn.Shape{10}, data)
	}

	return labels, nil
}

func download(uri, path string) error {
	resp, err := http.Get(uri)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	_, err = io.Copy(f, resp.Body)
	return err
}

func Load() (xTrain, yTrain, xTest, yTest []*nn.Tensor, err error) {
	if _, err := os.Stat(baseDir); os.IsNotExist(err) {
		if err := os.MkdirAll(baseDir, 0777); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	if _, err := os.Stat(trainImagePath); os.IsNotExist(err) {
		if err := download(url+trainImage, trainImagePath); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	if _, err := os.Stat(trainLabelPath); os.IsNotExist(err) {
		if err := download(url+trainLabel, trainLabelPath); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		if err := download(url+testImage, testImagePath); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	if _, err := os.Stat(testLabelPath); os.IsNotExist(err) {
		if err := download(url+testLabel, testLabelPath); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	xTrain, err = LoadImage(trainImagePath)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTrain, err = LoadLabel(trainLabelPath)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	xTest, err = LoadImage(testImagePath)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTest, err = LoadLabel(testLabelPath)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return xTrain, yTrain, xTest, yTest, nil
}
