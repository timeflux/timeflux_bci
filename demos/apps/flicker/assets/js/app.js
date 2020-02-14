'use strict';

let flicker = null;
let io = new IO();
let options = {};
let frequencies = {};
let predictions = {};
let n_predictions = 0;
let predict = false;
const calibrate_button = document.getElementById('calibrate');
const start_button = document.getElementById('start');

load_settings().then(settings => {
    options = settings.flicker;
    if (options.target_size) {
        document.documentElement.style.setProperty('--target-size', options.target_size);
    }
    calibrate_button.addEventListener('click', calibrate);
    start_button.addEventListener('click', classify);
});

async function calibrate() {
    document.body.classList.toggle('quiet');
    calibrate_button.classList.toggle('hidden');
    flicker = new Flicker(io, options);
    frequencies = flip(flicker.options.targets);
    await flicker.calibrate();
    document.body.classList.toggle('quiet');
    start_button.classList.toggle('hidden');
}

async function classify() {
    document.body.classList.toggle('quiet');
    start_button.classList.toggle('hidden');
    reset();
    flicker.start();
}

io.subscribe('events');
io.on('events', (data, meta) => {
    for (let timestamp in data) {
        try {
            if (data[timestamp]['label'] == 'predict') {
                data = JSON.parse(data[timestamp].data);
                prediction(data.result);
            }
        } catch(e) {}
    }
});

function reset() {
    for (let frequency in frequencies) {
        predictions[frequencies[frequency]] = 0;
    }
    n_predictions = 0;
    predict = true;
}

function prediction(frequency) {
    // Run only in prediction mode
    if (!predict) return;
    // Update
    predictions[frequencies[frequency]]++;
    n_predictions++;
    if (n_predictions >= options.min_predictions) {
        // We hit the minimum number of predictions
        let sorted = Object.keys(predictions).sort((a, b) => { return predictions[b] - predictions[a]});
        let target = sorted[0];
        if ((predictions[sorted[0]] - predictions[sorted[1]]) >= options.min_difference) {
            // There is enough difference between the first two candidates
            if (flicker.options.targets[target] == 0) {
                // The rest target was predicted
                // Just reset the predictions
                reset();
            } else {
                // Highlight the target and run the matching action
                predict = false;
                flicker.highlight(sorted[0]).then(() => {
                    reset();
                });
                action(target);
            }
        }
    }
}

function action(target) {
    console.log(target);
}

function flip(obj) {
  return Object.keys(obj).reduce((ret, key) => {
    ret[obj[key]] = key;
    return ret;
  }, {});
}