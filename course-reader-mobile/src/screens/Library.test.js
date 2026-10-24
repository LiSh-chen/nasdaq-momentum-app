import React from 'react';
import { Alert } from 'react-native';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Library from './Library';
import { getFavorites, getProgress } from '../store';

const course = (id, title) => ({ id, title, image: null, url: `https://x/${id}`, sourceCode: 's', sourceName: 'S', unified: false, levels: [], topics: [], description: '', instructors: [], features: [], free: true });

beforeEach(async () => {
  await AsyncStorage.clear();
  await AsyncStorage.setItem('favorites', JSON.stringify({ a: course('a', '課程甲') }));
  await AsyncStorage.setItem('progress', JSON.stringify({ b: { course: course('b', '課程乙'), index: 0, title: '第一章', total: 3, updatedAt: 1 } }));
});

test('編輯模式可移除收藏與閱讀紀錄，且不會開啟課程', async () => {
  const onOpen = jest.fn();
  const alert = jest.spyOn(Alert, 'alert').mockImplementation((t, m, buttons) => buttons.find((b) => b.style === 'destructive').onPress());
  const { findByText, getAllByText, queryByText } = render(<Library onOpenCourse={onOpen} />);
  await findByText('課程甲');
  expect(queryByText('移除')).toBeNull();

  fireEvent.press(await findByText('編輯'));
  fireEvent.press(getAllByText('課程甲')[0]);
  expect(onOpen).not.toHaveBeenCalled();

  fireEvent.press(getAllByText('移除')[1]); // 收藏
  await waitFor(async () => expect(await getFavorites()).toEqual({}));
  fireEvent.press(getAllByText('移除')[0]); // 閱讀紀錄
  await waitFor(async () => expect(await getProgress()).toEqual({}));
  alert.mockRestore();
});

test('不在編輯模式時點課程會開啟', async () => {
  const onOpen = jest.fn();
  const { findByText } = render(<Library onOpenCourse={onOpen} />);
  fireEvent.press(await findByText('課程甲'));
  expect(onOpen).toHaveBeenCalledTimes(1);
});
