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
const controls = document.getElementsByClassName('control');
const player = document.getElementById('player');
const player_title = document.getElementById('player_title');
const player_artist = document.getElementById('player_artist');
const player_album = document.getElementById('player_album');
const player_cover = document.getElementById('player_cover');
const audio = document.getElementById('audio');
const media = [
    'assets/media/01 Sweet Confusion.mp3',
    'assets/media/02 Tell Me.mp3',
    'assets/media/02 Strasbourg.mp3'
];
let meta = [];
let track = -1;


load_settings().then(settings => {
    options = settings.flicker;
    if (options.target_size) {
        document.documentElement.style.setProperty('--target-size', options.target_size);
    }
    calibrate_button.addEventListener('click', calibrate);
    start_button.addEventListener('click', start);
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

async function start() {
    document.body.classList.toggle('quiet');
    start_button.classList.toggle('hidden');
    await get_meta();
    player.classList.toggle('hidden');
    for (let control of controls) {
        control.classList.toggle('hidden');
    }
    next();
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
                toggle_control(target);
                flicker.highlight(target).then(() => {
                    toggle_control(target);
                    reset();
                });
                action(target);
            }
        }
    }
}

function action(target) {
    if (target == 'play') play();
    if (target == 'next') next();
}

async function get_meta() {
    for (let i in media) {
        let path = media[i];
        if (!path.startsWith('http')) path = window.location.href + path;
        await id3(path);
    }
}

async function id3(path) {
    return new Promise((resolve, reject) => {
        jsmediatags.read(path, {
            onSuccess: (id3) => {
                let info = {};
                info.title = 'title' in id3.tags ? id3.tags.title : '';
                info.artist = 'artist' in id3.tags ? id3.tags.artist : '';
                info.album = 'album' in id3.tags ? id3.tags.album : '';
                if ('picture' in id3.tags) {
                    const buf = id3.tags.picture.data;
                    const img = btoa(new Uint8Array(buf).reduce((data,byte)=>(data.push(String.fromCharCode(byte)),data),[]).join(''))
                    info.cover = 'data:' + id3.tags.picture.format + ';base64,' + img;
                } else {
                    info.cover =  null;
                }
                meta.push(info);
                resolve(info);
            },
            onError: (error) => {
                reject(error);
            }
        });
    });
}

function toggle_control(target) {
    for (let control of controls) {
        if (control.parentElement.id != target) {
            control.classList.toggle('hidden');
        }
    }
}

function play() {
    if (audio.paused) {
        audio.play();
    } else {
        audio.pause();
    }
}

function next() {
    track++;
    let autoplay = !audio.paused;
    if (track == media.length) track = 0;
    audio.src = media[track];
    player_cover.src = meta[track].cover;
    player_title.textContent = meta[track].title;
    player_artist.textContent = meta[track].artist;
    player_album.textContent = meta[track].album;
    if (autoplay) audio.play();
}

function flip(obj) {
  return Object.keys(obj).reduce((ret, key) => {
    ret[obj[key]] = key;
    return ret;
  }, {});
}