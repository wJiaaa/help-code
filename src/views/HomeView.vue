<template>
  <div class="page">
    <div class="page-title">
      <div class="page-title-text">Deep Learning</div>
      <img src="../assets/img/Index.gif" />
    </div>
    <HomeItem :list="list" title="Classical Deep Learning Network Model" />
    <HomeItem :list="applicationList" title="Application" />
  </div>
</template>
<script setup>
import HomeItem from '../components/HomeItem.vue'
const resolveImg = (url) => {
  return new URL(`../assets/img/${url}`, import.meta.url).href
}

const list = [
  {
    coverUrl: resolveImg('one/CNN.jpg'),
    title: 'CNN',
    desc: 'Simulates the convolution operation in biological neural networks to automatically extract hierarchical features from the input data.',
    isNew: true,
    key: 'cnn'
  },
  {
    coverUrl: resolveImg('one/Dot-product Self-Attention Networks.png'),
    title: 'Dot-product Self-Attention Networks',
    desc: 'Neural network structure for point product computation of self-attentive weights, capable of capturing long-range dependencies within the input sequence.',
    isNew: false
  },
  {
    coverUrl: resolveImg('one/GANs.jpg'),
    title: 'GANs',
    desc: 'Adversarial learning by training the generator and discriminator simultaneously, enabling the generator to produce realistic data samples.',
    isNew: true
  },
  {
    coverUrl: resolveImg('one/RBFNs.png'),
    title: 'RBFNs',
    desc: 'Enables nonlinear mapping and approximation in the input space by using radial basis functions as activation functions.',
    isNew: false
  },
  {
    coverUrl: resolveImg('one/ResNets.jpg'),
    title: 'ResNets',
    desc: 'Mitigate the gradient disappearance problem by introducing a residual module to enable deeper model training.',
    isNew: true
  },
  {
    coverUrl: resolveImg('one/RNNs.jpg'),
    title: 'RNNs',
    desc: 'By introducing circular connections in the network structure, the model is able to capture temporal information and long-range dependencies.',
    isNew: false
  }
]
const applicationList = [
  {
    coverUrl: resolveImg('two/OpenAI.jpg'),
    title: 'OpenAI',
    isNew: false,
    linkUrl: 'https://platform.openai.com/overview'
  },
  {
    coverUrl: resolveImg('two/RunwayML.jpg'),
    title: 'RunwayML',
    isNew: true,
    linkUrl: 'https://app.runwayml.com/video-tools/teams/zhuoqunxue/ai-tools'
  },
  {
    coverUrl: resolveImg('two/stability.jpg'),
    title: 'stability',
    isNew: false,
    linkUrl: 'https://stability.ai'
  },
  {
    coverUrl: resolveImg('two/Teachable Machine.jpg'),
    title: 'Teachable Machine',
    isNew: true,
    linkUrl: 'https://teachablemachine.withgoogle.com'
  }
]
</script>
<style lang="less" scoped>
.page {
  min-width: 1200px;
  background-color: #f1f3f4;
  .page-title {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 550px;
    font: 400 44px / 52px Google Sans, sans-serif;
    margin: 0;
    position: relative;
    .page-title-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #fff;
      z-index: 999;
    }
    img {
      filter: brightness(50%);
      grid-column: 1 / -1;
      grid-row: 1 / -1;
      height: 100%;
      object-fit: cover;
      width: 100%;
    }
  }
  .site-page {
    padding: 40px 0;
    .site-main {
      margin: 0 auto;
      padding: 0 24px;
      max-width: 1488px;
      .site-main-header {
        -webkit-box-align: center;
        -webkit-align-items: center;
        -moz-box-align: center;
        -ms-flex-align: center;
        align-items: center;
        text-align: center;
        .header-text {
          h2 {
            font-weight: 400;
            font-size: 44px;
            font-family: Google Scans;
            margin: 0;
          }
          p:first-child {
            margin-top: 0;
          }
          p {
            margin: 16px 0;
            font-size: 18px;
            font-family: Roboto;
          }
        }
      }
    }
    .site-body {
      display: grid;
      grid-gap: 24px;
      grid-auto-rows: auto;
      grid-auto-columns: auto;
      grid-auto-flow: row;
      margin: 32px auto 0;
      grid: auto-flow/repeat(3, 1fr);
      .site-item {
        background-color: #fff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        box-shadow: none;
        img {
          border: 0;
          // height: auto;
          // max-width: 100%;
          max-height: 256px;
          width: 100%;
          cursor: pointer;
        }
        .item-text {
          padding: 16px;
          .item-new {
            border-radius: 4px;
            margin-bottom: 16px;
            padding: 4px 8px;
            color: #fff;
            background-color: #1a73e8;
            font-weight: 500;
            font-size: 14px;
          }
          .item-title {
            cursor: pointer;
            font: 400 32px/40px Google Sans, Noto Sans, Noto Sans JP, Noto Sans KR,
              Noto Naskh Arabic, Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
            color: #1a73e8;
            margin-bottom: 20px;
            margin-top: 15px;
          }
        }
      }
    }
  }
}
</style>
