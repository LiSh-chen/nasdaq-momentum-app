import React from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';

export const colors = {
  bg: '#f3f4f6',
  card: '#ffffff',
  dark: '#111827',
  primary: '#2563eb',
  primarySoft: '#dbeafe',
  text: '#111827',
  muted: '#6b7280',
  line: '#e5e7eb',
};

export function Btn({ label, onPress, disabled, dark }) {
  return (
    <Pressable onPress={onPress} disabled={disabled} style={[ui.btn, disabled && { opacity: 0.35 }]}>
      <Text style={[ui.btnText, dark && { color: colors.primary }]}>{label}</Text>
    </Pressable>
  );
}

export function Chip({ label, active, onPress }) {
  return (
    <Pressable onPress={onPress} style={[ui.chip, active && ui.chipActive]}>
      <Text style={[ui.chipText, active && ui.chipTextActive]}>{label}</Text>
    </Pressable>
  );
}

export const ui = StyleSheet.create({
  btn: { paddingHorizontal: 10, paddingVertical: 8 },
  btnText: { color: '#fff', fontSize: 16 },
  chip: {
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16, marginRight: 8,
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line,
  },
  chipActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  chipText: { color: colors.text, fontSize: 13 },
  chipTextActive: { color: '#fff' },
});
