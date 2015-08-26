package main

import (
	"encoding/csv"
	"fmt"
	"os"
	// "runtime"
	"strconv"
	"time"

	"github.com/padster/eego/grading"
	"github.com/padster/eego/trees"
	"github.com/padster/go-sound/util"
)

type Channel struct {
	Id      string
	Samples []int
}

func main() {
	// runtime.GOMAXPROCS(2)

	gradeSubjectSeries(1, 1, 9)
}

func gradeSubjectSeries(subject int, trainSeries int, testSeries int) {
	// EEG_CHANNEL := "FC1"
	// EVENT_CHANNEL := "FirstDigitTouch"

	fmt.Printf("Loading training data...\n")
	data := loadData(subject, trainSeries, false)

	fmt.Printf("Loading training events...\n")
	events := loadEvents(1, 1)
	
	fmt.Printf("Training...\n")
	for _, vd := range data {
		for _, ve := range events {
			f := trees.NewForest(150, 1, 1000)
			f.Train(vd.Samples, ve.Samples)
			dId, eId := vd.Id, ve.Id
			if len(dId) > 4 {
				dId = dId[:4]
			}
			for len(dId) < 4 {
				dId = dId + "_"
			}
			if len(eId) > 7 {
				eId = eId[:7]
			}
			for len(eId) < 7 {
				eId = eId + "_"
			}

			fmt.Printf("%s\t%s\tV = %d\t~E = %f\n", 
				dId, eId, f.DecisionNodes(), f.AverageErrors())
			if f.DecisionNodes() > 1 {
				fmt.Printf("*******WOAH*******\n")
				fmt.Printf("*******WOAH*******\n")
				fmt.Printf("*******WOAH*******\n")
			}
		}
	}

	fmt.Printf("Trained!\n")
}

func channelSamples(channels []Channel, id string) []int {
	for _, c := range channels {
		if c.Id == id {
			return c.Samples
		}
	}
	panic("Cannot access unknown channel " + id + ".")
}

// verifies the AUC grades for some test cases.
func verifyAuc() {
	// TODO(padster): migrate to test suite
	fmt.Printf("3/4 == %f\n", grading.RocAucScore(
		[]int{0, 0, 1, 1},
		[]float64{0.1, 0.4, 0.35, 0.8},
	))
	fmt.Printf("1/3 == %f\n", grading.RocAucScore(
		[]int{0, 0, 0, 0, 1, 1, 1},
		[]float64{0.1, 0.6, 0.6, 0.23, 0.1, 0.23, 0.5},
	))
	fmt.Printf("0.7916.. == %f\n", grading.RocAucScore(
		[]int{1, 0, 1, 0, 1, 1, 1, 1},
		[]float64{0.8, 0.5, 0.44, 0.1, 0.2, 0.9, 0.9, 0.5},
	))
}

// loadData Loads EEG channel data for a given subject and series.
func loadData(subject int, series int, test bool) []Channel {
	var filename string
	if test {
		filename = fmt.Sprintf("data/test/subj%d_series%d_data.csv", subject, series)
	} else {
		filename = fmt.Sprintf("data/train/subj%d_series%d_data.csv", subject, series)
	}
	return loadChannels(filename)
}

// loadEvents loads event flags for a given subject and series.
func loadEvents(subject int, series int) []Channel {
	filename := fmt.Sprintf("data/train/subj%d_series%d_events.csv", subject, series)
	return loadChannels(filename)
}

// loadChannels loads the CSV into column-major array of channels.
func loadChannels(filename string) []Channel {
	fmt.Printf(" > Loading channels from %s\n", filename)
	if file, err := os.Open(filename); err == nil {
		defer file.Close()

		r := csv.NewReader(file)
		r.FieldsPerRecord = -1

		if data, err := r.ReadAll(); err == nil {
			channels := make([]Channel, len(data[0])-1, len(data[0])-1)
			for i, cid := range data[0] {
				if i != 0 {
					channels[i-1] = Channel{
						cid,
						make([]int, len(data)-1),
					}
				}
			}
			for i, row := range data {
				if i != 0 {
					for j, s := range row {
						if j != 0 {
							channels[j-1].Samples[i-1], _ = strconv.Atoi(s)
						}
					}
				}
			}
			fmt.Printf("%d channels loaded, with %d samples\n", len(channels), len(channels[0].Samples))
			return channels
		} else {
			panic(err)
		}
	} else {
		panic(err)
	}
}

// asUiChannel converts an array of values into a realtime(ish) channel of samples.
func asUiChannel(samples []int) <-chan float64 {
	min, max := minMax(samples)
	c := make(chan float64)
	go func() {
		for _, s := range samples {
			scaled := 2.00*float64(s-min)/float64(max-min) - 1.0
			c <- scaled
			time.Sleep(2 * time.Millisecond)
		}
	}()
	return c
}

// minMax returns the highest and lowest values in an array
func minMax(values []int) (int, int) {
	/*
		min, max := values[0], values[0]
		for _, v := range values {
			if v < min {
				min = v
			} else if v > max {
				max = v
			}
		}
		return min, max
	*/
	// NOTE(padster): some data has some really big extremes. This normalizes them to the same scale.
	return -1200, 3000
}

// asEventChannel converts an array of 0/1 events to an event at that time.
func asEventChannel(message string, events []Channel) <-chan interface{} {
	c := make(chan interface{})
	go func() {
		for i := 0; i < len(events[0].Samples); i++ {
			switch {
			case events[0].Samples[i] == 1:
				c <- util.Event{1.0, 0.0, 0.0}
			case events[1].Samples[i] == 1:
				c <- util.Event{1.0, 1.0, 0.0}
			case events[2].Samples[i] == 1:
				c <- util.Event{0.0, 1.0, 0.0}
			case events[3].Samples[i] == 1:
				c <- util.Event{0.0, 1.0, 1.0}
			case events[4].Samples[i] == 1:
				c <- util.Event{0.0, 0.0, 1.0}
			case events[5].Samples[i] == 1:
				c <- util.Event{1.0, 0.0, 1.0}
			default:
				c <- nil
			}
			time.Sleep(2 * time.Millisecond)
		}
	}()
	return c
}
