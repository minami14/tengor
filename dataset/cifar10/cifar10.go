// https://www.cs.toronto.edu/~kriz/cifar.html

package cifar10

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/minami14/tengor/nn"
)

const (
	url      = "https://www.cs.toronto.edu/~kriz/"
	filename = "cifar-10-binary.tar.gz"
	h        = 32
	w        = 32
	c        = 3
	size     = h * w * c
)

var (
	basedir  = "tengor/dataset/cifar10/"
	filepath = basedir + filename
)

func init() {
	cache, err := os.UserCacheDir()
	if err != nil {
		return
	}

	basedir = cache + "/" + basedir
	filepath = basedir + filename
}

func download() error {
	if _, err := os.Stat(basedir); os.IsNotExist(err) {
		if err := os.MkdirAll(basedir, 0777); err != nil {
			return err
		}
	}

	resp, err := http.Get(url + filename)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	_, err = io.Copy(f, resp.Body)
	return err
}

func read(x, y []*nn.Tensor, reader io.Reader) error {
	for i := 0; i < 10000; i++ {
		buf := make([]byte, size)
		if _, err := reader.Read(buf[:1]); err != nil {
			return err
		}

		yRaw := make([]float64, 10)
		yRaw[buf[0]] = 1
		y[i] = nn.TensorFromSlice(nn.Shape{10}, yRaw)

		for start := 0; start < size; {
			n, err := reader.Read(buf[start:size])
			if err != nil && err != io.EOF {
				return err
			}
			start += n
		}

		xRaw := make([]float64, size)
		for j := 0; j < size; j++ {
			xRaw[j] = float64(buf[j]) / 255
		}

		x[i] = nn.TensorFromSlice(nn.Shape{h, w, c}, xRaw)
	}

	return nil
}

// Load downloads and loads cifar10 dataset.
func Load() (xTrain, yTrain, xTest, yTest []*nn.Tensor, err error) {
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		if err := download(); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	f, err := os.Open(filepath)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer func() { _ = f.Close() }()

	g, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	xTrain = make([]*nn.Tensor, 50000)
	yTrain = make([]*nn.Tensor, 50000)
	xTest = make([]*nn.Tensor, 10000)
	yTest = make([]*nn.Tensor, 10000)
	reader := tar.NewReader(g)
	for {
		header, err := reader.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, nil, nil, err
		}

		if !strings.Contains(header.Name, ".bin") {
			continue
		}

		switch header.Name {
		case "cifar-10-batches-bin/data_batch_1.bin":
			if err := read(xTrain[:10000], yTrain[:10000], reader); err != nil {
				return nil, nil, nil, nil, err
			}
		case "cifar-10-batches-bin/data_batch_2.bin":
			if err := read(xTrain[10000:20000], yTrain[10000:20000], reader); err != nil {
				return nil, nil, nil, nil, err
			}
		case "cifar-10-batches-bin/data_batch_3.bin":
			if err := read(xTrain[20000:30000], yTrain[20000:30000], reader); err != nil {
				return nil, nil, nil, nil, err
			}
		case "cifar-10-batches-bin/data_batch_4.bin":
			if err := read(xTrain[30000:40000], yTrain[30000:40000], reader); err != nil {
				return nil, nil, nil, nil, err
			}
		case "cifar-10-batches-bin/data_batch_5.bin":
			if err := read(xTrain[40000:50000], yTrain[40000:50000], reader); err != nil {
				return nil, nil, nil, nil, err
			}
		case "cifar-10-batches-bin/test_batch.bin":
			if err := read(xTest, yTest, reader); err != nil {
				return nil, nil, nil, nil, err
			}
		}
	}

	return xTrain, yTrain, xTest, yTest, nil
}
