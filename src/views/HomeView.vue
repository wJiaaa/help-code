<template>
  <div class="page">
    <div class="page-title">
      <div class="page-title-text">Deep Learning</div>
      <video
        src="https://developers.google.com/machine-learning/crash-course/images/landing-hero-video.webm"
      />
    </div>
    <HomeItem
      :list="list"
      title="Classical Deep Learning Network Model"
      :descList="[
        'The foundational courses cover machine learning fundamentals and core concepts.',
        'We recommend taking them in the order below.'
      ]"
    />
    <HomeItem
      :list="applicationList"
      title="Application"
      :descList="[
        'The foundational courses cover machine learning fundamentals and core concepts.',
        'We recommend taking them in the order below.'
      ]"
    />
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
    desc: 'A brief introduction to machine learning.',
    isNew: true,
    key: 'cnn'
  },
  {
    coverUrl: resolveImg('one/Dot-product Self-Attention Networks.png'),
    title: 'Dot-product Self-Attention Networks',
    desc: 'A hands-on course to explore the critical basics of machine learning.',
    isNew: false
  },
  {
    coverUrl: resolveImg('one/GANs.jpg'),
    title: 'GANs',
    desc: 'A brief introduction to machine learning.',
    isNew: true
  },
  {
    coverUrl: resolveImg('one/RBFNs.png'),
    title: 'RBFNs',
    desc: 'A hands-on course to explore the critical basics of machine learning.',
    isNew: false
  },
  {
    coverUrl: resolveImg('one/ResNets.jpg'),
    title: 'ResNets',
    desc: 'A brief introduction to machine learning.',
    isNew: true
  },
  {
    coverUrl: resolveImg('one/RNNs.jpg'),
    title: 'RNNs',
    desc: 'A hands-on course to explore the critical basics of machine learning.',
    isNew: false
  }
]
const applicationList = [
  {
    coverUrl: resolveImg('two/OpenAI.jpg'),
    title: 'OpenAI',
    desc: 'A brief introduction to machine learning.',
    isNew: false
  },
  {
    coverUrl: resolveImg('two/RunwayML.jpg'),
    title: 'RunwayML',
    desc: 'A brief introduction to machine learning.',
    isNew: true
  },
  {
    coverUrl: resolveImg('two/stability.jpg'),
    title: 'stability',
    desc: 'A brief introduction to machine learning.',
    isNew: false
  },
  {
    coverUrl: resolveImg('two/Teachable Machine.jpg'),
    title: 'Teachable Machine',
    desc: 'A brief introduction to machine learning.',
    isNew: true
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
    video {
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
