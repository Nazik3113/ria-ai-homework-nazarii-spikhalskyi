const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");
const tfnode = require("@tensorflow/tfjs-node");
const util = require('util');

const readdir = util.promisify(fs.readdir);
const readFile = util.promisify(fs.readFile);

class AI {
    async loadData(baseDir="./data/seg_train/seg_train") {
        const folders = await readdir(baseDir);
        const allFiles = [];
        for (const i in folders) {
            const folder = folders[i];
            const files = await readdir(path.join(baseDir, folder));
            allFiles.push(...files.map(f => {
                return [
                    path.join(baseDir, folder, f),
                    i
                ]
            }));
        }
    
        let images = [];
        let labels = [];
        for (const [file, i] of allFiles) {
            let buffer = await readFile(file);
            let tfimage = tfnode.node.decodeImage(buffer, 3);
            tfimage = tf.image.resizeBilinear(tfimage, [28, 28]),
            tfimage = tfimage.cast("float32").div(255);
            images.push(tfimage);
            labels.push(i);
        }
        return [
            tf.stack(images),
            tf.oneHot(tf.tensor1d(labels, 'int32'), 6)
        ] 
    }

    setupModel() {
        const model = tf.sequential();

        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 3],
            filters: 32,
            kernelSize: 3,
            activation: "relu"
        }));

        model.add(tf.layers.dense({
            inputShape: [1],
            units: 1,
            activation: "relu",
            kernelInitializer: "ones"
        }));

        model.add(tf.layers.maxPool2d({
            poolSize: [2, 2],
            strides: [1, 1]
        }));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: 6,
            activation: "softmax"
        }));

        model.compile({
            optimizer: tf.train.sgd(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ["accuracy"]
        });

        return model;
    }

    async trainAndSaveModel() {
        const [xs, ys] = await this.loadData();
        const model = this.setupModel();

        await model.fit(xs, ys, {
            epochs: 1000,
            batchSize: 6000
        });        

        await model.save("file:///Users/nazik/Desktop/ria-AI-homework/model");
    }

    run() {
        this.trainAndSaveModel();
    }
}

const ai = new AI();
ai.run();