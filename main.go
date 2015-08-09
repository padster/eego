package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/padster/go-sound/util"
)

type Channel struct {
	Id      string
	Samples []int
}

func main() {
	runtime.GOMAXPROCS(2)

	inputs := loadData(3, 8, false)
	min, max := 10000, -10000
	for _, s := range inputs[0].Samples {
		if min > s {
			min = s
		}
		if max < s {
			max = s
		}
	}

	// Renders the EEG data for one of the channels to screen:
	s := util.NewScreen(1600, 400, 1)
	s.Render(asUiChannel(inputs[0].Samples), 1)
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
	c := make(chan float64)
	go func() {
		for _, s := range samples {
			// HACK - input range is: -1404 - 1032 for (3, 8, 0)
			lb, ub := -1404, 1032
			scaled := 2.00*float64(s-lb)/float64(ub-lb) - 1.0
			c <- scaled
			time.Sleep(2 * time.Millisecond)
		}
	}()
	return c
}
