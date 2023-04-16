<template>
  <section class="site-page">
    <div class="site-main">
      <header class="site-main-header">
        <div class="header-text">
          <h2>{{ title }}</h2>
          <div v-for="(item, index) in descList" :key="index">
            <p>{{ item }}</p>
          </div>
        </div>
      </header>
      <div class="site-body">
        <div class="site-item" v-for="(item, index) in list" :key="index">
          <div @click="goRoute(item)">
            <img :src="item.coverUrl" />
          </div>
          <div class="item-text">
            <span class="item-new" v-if="item.isNew">NEW</span>
            <div class="item-title">
              <span @click="goRoute(item)">{{ item.title }}</span>
            </div>
            <div class="item-desc">{{ item.desc }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
<script setup>
import { useRouter } from 'vue-router'
const router = useRouter()
defineProps({
  list: {
    type: Array,
    default: () => []
  },
  title: {
    type: String,
    default: ''
  },
  descList: {
    type: Array,
    default: () => []
  }
})
const goRoute = (item) => {
  if (item.key === 'cnn') {
    router.push({ path: '/explore', query: { isFromCnnClick: true } })
  }
}
</script>
<style lang="less" scoped>
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
          font: 400 32px/40px Google Sans, Noto Sans, Noto Sans JP, Noto Sans KR, Noto Naskh Arabic,
            Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
          color: #1a73e8;
          margin-bottom: 20px;
          margin-top: 15px;
          span {
            cursor: pointer;
          }
        }
      }
    }
  }
}
</style>
