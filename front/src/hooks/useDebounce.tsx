// src/hooks/useDebounce.ts
import { useState, useEffect } from 'react';

// T는 어떤 타입이든 될 수 있다는 의미의 제네릭입니다.
export function useDebounce<T>(value: T, delay: number): T {
  // 디바운싱된 값을 저장할 state
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(
    () => {
      // value가 변경되면, delay 이후에 debouncedValue를 업데이트하는 타이머를 설정합니다.
      const handler = setTimeout(() => {
        setDebouncedValue(value);
      }, delay);

      // 이 useEffect가 다시 실행되기 전(즉, value가 다시 바뀌기 전)에
      // 이전 타이머를 제거합니다. 이렇게 함으로써 타이핑이 계속되는 동안에는
      // debouncedValue가 업데이트되지 않습니다.
      return () => {
        clearTimeout(handler);
      };
    },
    [value, delay] // value나 delay가 변경될 때만 이 effect를 다시 실행합니다.
  );

  return debouncedValue;
}