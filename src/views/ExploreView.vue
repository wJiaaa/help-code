<template>
  <div class="main-page">
    <NavBar v-model:activeItem="activeItem" />
    <div>321</div>
  </div>
</template>
<script setup>
import { ref, watch } from 'vue'
import NavBar from '../components/NavBar.vue'
const activeItem = ref()

watch(
  () => activeItem.value,
  (val) => {
    dealShowInfo(val)
  },
  {
    deep: true
  }
)
// 根据点击的不同元素处理不同的情况
const dealShowInfo = (item) => {
  console.log('val', item)
  if (item === 'Prerequisites') {
    console.log(2)
  } else {
    if (item.clickItemInfoId === 1) {
      // 显示inintroduction
      console.log(1)
    } else {
      // 显示 check you understand
      console.log(3)
    }
  }
}
</script>
<style lang="less" scoped>
.main-page {
  height: calc(100% - 46px);
  display: flex;
}
</style>
