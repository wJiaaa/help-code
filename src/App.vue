<script setup>
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { ref, watch } from 'vue'
const route = useRoute()
const current = ref(['Explore'])

watch(
  () => route.name,
  (val) => {
    if (val) {
      current.value = [val]
    }
  }
)
</script>

<template>
  <a-menu v-model:selectedKeys="current" mode="horizontal">
    <a-menu-item key="Home"> <RouterLink to="/">Home</RouterLink> </a-menu-item>
    <a-menu-item key="Explore"> <RouterLink to="/explore">Explore</RouterLink> </a-menu-item>
  </a-menu>
  <RouterView />
</template>
<style lang="less" scoped></style>
