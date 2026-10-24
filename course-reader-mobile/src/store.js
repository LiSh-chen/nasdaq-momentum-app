import AsyncStorage from '@react-native-async-storage/async-storage';

const read = async (key) => {
  try {
    return JSON.parse(await AsyncStorage.getItem(key)) ?? {};
  } catch {
    return {};
  }
};
const write = (key, value) => AsyncStorage.setItem(key, JSON.stringify(value));

const slim = (c) => ({
  id: c.id, title: c.title, image: c.image, url: c.url,
  sourceCode: c.sourceCode, sourceName: c.sourceName, unified: c.unified,
  levels: c.levels, topics: c.topics, description: c.description, instructors: c.instructors,
  features: c.features, free: c.free,
});

export const getFavorites = () => read('favorites');

export async function toggleFavorite(course) {
  const favs = await read('favorites');
  if (favs[course.id]) delete favs[course.id];
  else favs[course.id] = slim(course);
  await write('favorites', favs);
  return Boolean(favs[course.id]);
}

export const getProgress = () => read('progress');

export async function saveProgress(course, { index, title, url, total }) {
  const all = await read('progress');
  all[course.id] = { course: slim(course), index, title, url, total, updatedAt: Date.now() };
  await write('progress', all);
}

export async function removeFavorite(id) {
  const favs = await read('favorites');
  delete favs[id];
  await write('favorites', favs);
}

export async function removeProgress(id) {
  const all = await read('progress');
  delete all[id];
  await write('progress', all);
}
