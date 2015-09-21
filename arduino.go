package main

import (
  "fmt"
  "log"
  "io/ioutil"
  "math"
  "runtime"
  "strings"
  "time"

  "github.com/tarm/serial"
  s "github.com/padster/go-sound/sounds"
  "github.com/padster/go-sound/output"
)

const (
  nsToSeconds        = 1e-9
  nsPerCycle         = s.SecondsPerCycle * 1e9
  outputSampleBuffer = 1 // how many output samples are written in the same loop
  tickerDuration     = time.Duration(outputSampleBuffer) * s.DurationPerCycle
  hzC                = 523.25
)

type Player struct {
  currentValue float64
  started bool
  running bool
}

// findArduino looks for the file that represents the Arduino
// serial connection. Returns the fully qualified path to the
// device if we are able to find a likely candidate for an
// Arduino, otherwise an empty string if unable to find
// something that 'looks' like an Arduino device.
func findArduino() string {
  contents, _ := ioutil.ReadDir("/dev")

  // Look for what is mostly likely the Arduino device
  for _, f := range contents {
    if strings.Contains(f.Name(), "tty.usbserial") ||
      strings.Contains(f.Name(), "ttyUSB") ||
      strings.Contains(f.Name(), "ttyACM") {
        fmt.Printf("Arduino serial at %s\n", "/dev/" + f.Name())
      return "/dev/" + f.Name()
    }
  }

  // Have not been able to find a USB device that 'looks'
  // like an Arduino.
  return ""
}

func main() {
  runtime.GOMAXPROCS(2)

  fmt.Printf("Open the serial cable...\n")
  port, err := serial.OpenPort(&serial.Config{Name: findArduino(), Baud: 9600})
  if err != nil {
    log.Fatal(err)
  }
  time.Sleep(1 * time.Second)

  fmt.Printf("Generate the tone definition...\n")
  player := &Player{}
  toPlay := s.SumSounds(
    s.NewHzFromChannel(player.sampledToneGenerator()),
    s.NewSineWave(hzC / 2.0),
  )

  buf := make([]byte, 128)
  startTime, readCount := time.Now(), 0
  for {
    if _, err := port.Read(buf); err != nil {
      if readCount == 0 {
        startTime = time.Now()
        player.Start(toPlay)
      }
      readCount++
  
      player.currentValue = float64(buf[0]) / 256.0
      if readCount % 100000 == 0 {
        fmt.Printf("Value = %f\n", player.currentValue)
      }
      if readCount % 1000000 == 0 {
        seconds := time.Since(startTime).Seconds()
        fmt.Printf("Read %d in %f seconds, at a rate of %f Hz\n", 
          readCount, seconds, float64(readCount) / seconds)
      }
    }
  }
}

// Start the player by initializing state and playing the tone
func (player *Player) Start(sound s.Sound) {
  fmt.Printf("Player starting...\n")

  player.running = true
  player.started = true
  go func() {
    output.Play(sound)
  }()
}

// Generate a stream of tone values at the correct sample rate, 
// based off player.currentValue as [0, 1]
func (player *Player) sampledToneGenerator() <-chan float64{
  samples := make(chan float64)

  go func() {
    player.running = true
    atNano := float64(time.Now().UnixNano())

    ticker := time.NewTicker(tickerDuration)
    defer ticker.Stop()

    for now := range ticker.C {
      nowNano := float64(now.UnixNano())

      for ; atNano < nowNano && player.running; atNano += nsPerCycle {
        currentValue := player.currentValue
        if !player.started {
          samples <- 0
        } else if player.running {
          // Snap to tones in a C major scale.
          toneOffset := int(currentValue * 8)
          toneValue := []int{0, 2, 4, 5, 7, 9, 11, 12}[toneOffset]
          currentSemitone := math.Pow(2.0, float64(toneValue) / 12.0)
          samples <- hzC * currentSemitone
        }
      }

      if !player.running { 
        break
      }
    }
    close(samples)
  }()

  return samples
}
