<template>
  <div>
    <div class="question-item" v-for="(item, index) in newList" :key="index">
      <div class="title">{{ item.question }}</div>
      <div
        class="answer"
        v-for="(answerItem, index) in item.answer"
        :key="index"
        @click="clickAnswer(item.rightOptionId, answerItem.id, answerItem)"
      >
        <div class="answer-main">
          <div class="answer-label">
            <span> {{ answerItem.option }}:</span>
            <span style="margin-left: 5px">{{ answerItem.label }}</span>
          </div>
          <svg
            v-if="answerItem.unChoose"
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
          >
            <path
              fill="currentColor"
              d="M12 14.975q-.2 0-.388-.075t-.312-.2l-4.6-4.6q-.275-.275-.275-.7t.275-.7q.275-.275.7-.275t.7.275l3.9 3.9l3.9-3.9q.275-.275.7-.275t.7.275q.275.275.275.7t-.275.7l-4.6 4.6q-.15.15-.325.213t-.375.062Z"
            />
          </svg>
          <template v-else>
            <svg
              v-if="answerItem.chooseRight"
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
            >
              <path
                fill="#137333"
                d="m9.55 18l-5.7-5.7l1.425-1.425L9.55 15.15l9.175-9.175L20.15 7.4L9.55 18Z"
              />
            </svg>
            <svg
              v-if="!answerItem.chooseRight"
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
            >
              <path
                fill="none"
                stroke="#d32f2f"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0m2.7-6.3l12.6 12.6"
              />
            </svg>
          </template>
        </div>
        <div class="answer-right" v-if="!answerItem.unChoose && answerItem.chooseRight">
          Correct answer.
        </div>
        <div class="try-again" v-if="!answerItem.unChoose && !answerItem.chooseRight">
          Try again.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  list: {
    type: Array,
    default: () => []
  }
})

const newList = ref(
  props.list.map((item) => {
    return {
      ...item,
      answer: item.answer.map((anItem) => {
        return {
          ...anItem,
          unChoose: true
        }
      })
    }
  })
)
/**
 * @Description 点击答案
 * @param questionAnswerId 当前题目答案
 * @param answerId 当前选择答案
 * @param answerItem 当前题目
 */
const clickAnswer = (questionAnswerId, answerId, answerItem) => {
  if (answerItem.unChoose) {
    answerItem.unChoose = false
    answerItem.chooseRight = questionAnswerId === answerId
  }
}
</script>
<style scoped lang="less">
.question-item {
  border: 1px solid #ccc;
  border-radius: 5px;
  margin-bottom: 20px;
  .title {
    background-color: #e8eaed;
    color: #202124;
    padding: 16px 24px;
    font: 500 14px/20px Roboto, Noto Sans, Noto Sans JP, Noto Sans KR, Noto Naskh Arabic,
      Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
  }
  .answer {
    border-top: 1px solid #ccc;
    padding: 16px 24px;
  }
  .answer-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .answer:hover {
    color: #1a73e8;
    cursor: pointer;
  }
  .answer-right {
    color: #137333;
    font-size: 14px;
    margin-top: 10px;
  }
  .try-again {
    color: #d32f2f;
    font-size: 14px;
    margin-top: 10px;
  }
}
</style>
