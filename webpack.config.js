const path = require('path')

module.exports = [
  // UMD bundle for browser
  {
    entry: './src/index.js',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'stoch.min.js',
      library: {
        name: 'stoch',
        type: 'umd'
      },
      globalObject: 'this'
    },
    externals: {
      '@tensorflow/tfjs': {
        commonjs: '@tensorflow/tfjs',
        commonjs2: '@tensorflow/tfjs',
        amd: '@tensorflow/tfjs',
        root: 'tf'
      }
    },
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader'
          }
        }
      ]
    }
  },
  // CJS bundle for Node.js
  {
    entry: './src/index.js',
    target: 'node',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'stoch.cjs.js',
      library: {
        type: 'commonjs2'
      }
    },
    externals: {
      '@tensorflow/tfjs': '@tensorflow/tfjs'
    },
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader'
          }
        }
      ]
    }
  }
]
