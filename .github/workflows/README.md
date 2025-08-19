# GitHub Actions 工作流程

## 错误修复说明

### 原问题
```
remote: Permission to RAG4SE/deepwiki-cli.git denied to github-actions[bot].
fatal: unable to access 'https://github.com/RAG4SE/deepwiki-cli/': The requested URL returned error: 403
Error: Process completed with exit code 128.
```

### 解决方案

1. **添加权限配置**：
   ```yaml
   permissions:
     contents: write  # 允许创建发布和推送标签
     packages: write  # 允许发布包
     pull-requests: read  # 允许读取PR信息
   ```

2. **使用正确的token认证**：
   - 在checkout步骤中使用`${{ secrets.GITHUB_TOKEN }}`
   - 在git push命令中使用HTTPS认证：
     ```bash
     git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git "v$VERSION"
     ```

3. **确保GitHub Release使用正确的token**：
   ```yaml
   token: ${{ secrets.GITHUB_TOKEN }}
   ```

### 工作流程功能

该工作流程在以下情况下自动运行：
- 推送到 `main` 或 `master` 分支
- `pyproject.toml` 文件发生变化
- 手动触发

工作流程会：
1. 检查版本是否发生变化
2. 检查版本是否已存在于PyPI
3. 构建和测试包
4. 发布到TestPyPI和PyPI
5. 创建Git标签和GitHub Release

### 所需的Secrets

确保在GitHub仓库的Settings > Secrets中配置：
- `PYPI_API_TOKEN`: PyPI的API token
- `TEST_PYPI_API_TOKEN`: TestPyPI的API token (可选)
- `GITHUB_TOKEN`: 自动提供，无需手动配置

### 权限要求

工作流程需要以下权限：
- `contents: write` - 创建标签和发布
- `packages: write` - 发布包
- `pull-requests: read` - 读取PR信息
