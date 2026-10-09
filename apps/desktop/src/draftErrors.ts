import type { Translate } from './models';

export type DraftErrorOperation = 'save' | 'delete' | 'rename' | 'copy' | 'load';
export function draftErrorTitle(operation: DraftErrorOperation, t: Translate): string {
  switch (operation) {
    case 'delete': return t('草稿删除异常', 'Draft deletion issue');
    case 'rename': return t('草稿重命名异常', 'Draft rename issue');
    case 'copy': return t('草稿复制异常', 'Draft copy issue');
    case 'load': return t('草稿载入异常', 'Draft loading issue');
    default: return t('草稿保存异常', 'Draft save issue');
  }
}
