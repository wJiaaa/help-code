<template>
  <nav class="nav">
    <ul class="quick-links">
      <li class="li-title">
        <div class="title-text">Quick Links</div>
      </li>
      <li :class="[activeItem === 'Prerequisites' ? 'activeLi' : '']">
        <div class="item-title hover title-style" @click="() => changeActiveItem()">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
            <path
              fill="none"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 6h16M4 12h16M4 18h7"
            />
          </svg>
          <div class="ml10">Prerequisites</div>
        </div>
      </li>
    </ul>
    <!-- cnn -->
    <ul class="links">
      <li class="li-title">
        <div class="title-text mt10">CNN</div>
      </li>
      <li class="links-item" v-for="(item, index) in cnnList" :key="index">
        <div class="title hover active-color title-style" @click="item.show = !item.show">
          <svg
            v-if="item.show"
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
          >
            <g transform="rotate(90 12 12)">
              <path fill="currentColor" d="M8 19V5l11 7l-11 7Z" />
            </g>
          </svg>
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
            <path fill="currentColor" d="M8 19V5l11 7l-11 7Z" />
          </svg>
          <div class="ml5">{{ item.title }}</div>
        </div>
        <ul v-show="item.show">
          <li
            :class="['links-item', activeItem === infoItem.id ? 'activeLi' : '']"
            v-for="(infoItem, index) in item.infoList"
            :key="index"
            @click="clickInfoItem(infoItem)"
          >
            <div class="title hover pl30 title-style">
              <svg
                v-if="infoItem.type === 1"
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
              >
                <path
                  fill="currentColor"
                  d="m10 15l5.19-3L10 9v6m11.56-7.83c.13.47.22 1.1.28 1.9c.07.8.1 1.49.1 2.09L22 12c0 2.19-.16 3.8-.44 4.83c-.25.9-.83 1.48-1.73 1.73c-.47.13-1.33.22-2.65.28c-1.3.07-2.49.1-3.59.1L12 19c-4.19 0-6.8-.16-7.83-.44c-.9-.25-1.48-.83-1.73-1.73c-.13-.47-.22-1.1-.28-1.9c-.07-.8-.1-1.49-.1-2.09L2 12c0-2.19.16-3.8.44-4.83c.25-.9.83-1.48 1.73-1.73c.47-.13 1.33-.22 2.65-.28c1.3-.07 2.49-.1 3.59-.1L12 5c4.19 0 6.8.16 7.83.44c.9.25 1.48.83 1.73 1.73Z"
                />
              </svg>
              <svg
                v-else
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
              >
                <path
                  fill="currentColor"
                  d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10s10-4.48 10-10S17.52 2 12 2zM9.29 16.29L5.7 12.7a.996.996 0 1 1 1.41-1.41L10 14.17l6.88-6.88a.996.996 0 1 1 1.41 1.41l-7.59 7.59a.996.996 0 0 1-1.41 0z"
                />
              </svg>
              <div class="ml5">{{ infoItem.title }}</div>
            </div>
          </li>
        </ul>
      </li>
    </ul>
    <!-- activity -->
    <ul class="links">
      <li class="li-title">
        <div class="title-text mt10">Activity</div>
      </li>
      <li
        :class="['links-item', activeItem === item.id ? 'activeLi' : '']"
        v-for="(item, index) in activetyList"
        :key="index"
        @click="clickInfoItem(item)"
      >
        <div class="title hover title-style">
          <div class="title-text">{{ item.title }}</div>
        </div>
      </li>
    </ul>
  </nav>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const emits = defineEmits(['update:activeItem'])

const cnnList = ref([
  {
    id: 1,
    title: 'Introduction to CNN (15 min)',
    show: false,
    infoList: [
      { id: 11, title: 'Introduction', type: 1 },
      { id: 12, title: 'Check Your Understanding', type: 2 }
    ]
  },
  {
    id: 2,
    title: 'Components of CNN (20 min)',
    show: false,
    infoList: [
      { id: 21, title: 'Introduction', type: 1 },
      { id: 22, title: 'Check Your Understanding', type: 2 }
    ]
  },
  {
    id: 3,
    title: 'CNN architecture (25 min)',
    show: false,
    infoList: [
      { id: 31, title: 'Introduction', type: 1 },
      { id: 32, title: 'Check Your Understanding', type: 2 }
    ]
  },
  {
    id: 4,
    title: 'Model and hyperparameters (60 min)',
    show: false,
    infoList: [
      { id: 41, title: 'Introduction', type: 1 },
      { id: 42, title: 'Check Your Understanding', type: 2 }
    ]
  },
  {
    id: 5,
    title: 'Design and Customize (11 min)',
    show: false,
    infoList: [
      { id: 51, title: 'Introduction', type: 1 },
      { id: 52, title: 'Check Your Understanding', type: 2 }
    ]
  }
])

const activetyList = ref([
  {
    id: 6,
    title: 'MCQs of Concepts (15 min)'
  },
  {
    id: 7,
    title: 'Training and Testing (20 min)'
  },
  {
    id: 8,
    title: 'Facial recognition (25 min)'
  }
])

const activeItem = ref('Prerequisites')

const clickInfoItem = (infoItem) => {
  activeItem.value = infoItem.id
  cnnList.value = cnnList.value.map((item) => {
    let flag = item.show
    item.infoList.map((arg) => {
      if (arg.id === infoItem.id) {
        flag = true
      }
    })
    return {
      ...item,
      show: flag
    }
  })
  emits('update:activeItem', infoItem.id)
}

const changeActiveItem = (item) => {
  activeItem.value = 'Prerequisites'
  emits('update:activeItem', item || activeItem.value)
}

if (route.query.isFromCnnClick) {
  activeItem.value = 'Prerequisites'
  emits('update:activeItem', activeItem.value)
}

defineExpose({
  changeActiveItem,
  clickInfoItem
})
</script>
<style scoped lang="less">
.ml10 {
  margin-left: 10px;
}
.ml5 {
  margin-left: 5px;
}
.mt10 {
  margin-top: 10px;
}
.pl30 {
  padding-left: 30px;
}
.li-title {
  color: rgba(0, 0, 0, 0.65);
  font-weight: 700;
}
li {
  line-height: 24px;
  margin: 5px 0;
  font-size: 13px;
  padding-left: 5px;
}
.li-title:hover {
  background: none;
  cursor: default;
}

.title-text {
  display: flex;
  align-items: center;
  padding-left: 24px;
}
.hover:hover {
  width: 100%;
  background-color: #f1f3f4;
  border-radius: 0 12px 12px 0;
  cursor: pointer;
  line-height: 24px;
}
.activeLi {
  .title-style {
    width: 100%;
    color: rgb(26, 115, 232);
    background-color: rgb(232, 240, 254);
    border-radius: 0 12px 12px 0;
  }
}
.links-item {
  .title {
    display: flex;
    align-items: center;
  }
}
// 用来修改点击时切换颜色
.active-color:active {
  color: rgb(26, 115, 232);
}
.links {
  padding-right: 10px;
}
.nav {
  height: 100%;
  width: 280px;
  border-right: 1px solid #ccc;
  .quick-links {
    font-size: 14px;
    border-bottom: 1px solid #ccc;
    padding-top: 15px;
    padding-bottom: 5px;
    padding-right: 10px;

    .item-title {
      display: flex;
      align-items: center;
      padding-left: 24px;
    }
  }
}
</style>
