#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
import time
import pandas as pd

def extract_note_id(note_link):
    """
    从小红书笔记链接中提取笔记ID。
    支持 https://www.xiaohongshu.com/explore/<note_id>[?...] 格式。
    """
    m = re.search(r'/explore/([^/?]+)', note_link)
    if m:
        return m.group(1)
    raise ValueError(f"无法从链接中提取笔记ID: {note_link}")

def fetch_comments(note_link, headers, page_size=20, pause=1):
    """
    根据笔记链接抓取评论列表，返回指定字段的字典列表。
    """
    note_id = extract_note_id(note_link)
    url = 'https://edith.xiaohongshu.com/api/sns/v1/note/comment/list'
    all_comments = []
    cursor = ""

    while True:
        payload = {
            "note_id": note_id,
            "page_size": page_size,
            "cursor": cursor,
            "top_comment_id": "",
            "image_formats": ["jpg", "webp"],
            "channel_id": "homefeed.recommend"
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # ---- debug: 打印 data['data'] 的 keys，确认翻页字段 ----
        if 'data' in data:
            print(f"  返回字段: {list(data['data'].keys())}")
        else:
            print(f"  Warning: 没有 data 字段, 返回内容: {data}")

        if not data.get('success', True) or 'data' not in data:
            print(f"[{note_link}] 接口错误: {data.get('message', '未知错误')}")
            break

        comments = data['data'].get('comments', [])
        # 有的接口可能用 next_cursor，这里两者都试一下
        cursor = data['data'].get('cursor') or data['data'].get('next_cursor','')

        for c in comments:
            user = c.get('user_info') or {}
            ts = c.get('insert_time', 0)
            # 如果是毫秒级时间戳，就除以 1000
            if ts > 1e12:
                ts = ts / 1000
            comment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

            row = {
                '评论ID':             c.get('id', ''),
                '笔记链接':           note_link,
                '笔记ID':             note_id,
                '用户链接':           f"https://www.xiaohongshu.com/user/profile/{user.get('user_id','')}",
                '用户ID':             user.get('user_id',''),
                '用户名称':           user.get('nickname',''),
                '评论内容':           c.get('content',''),
                '评论图片':           (c.get('images') or [{}])[0].get('url',''),
                '评论时间':           comment_time,
                '点赞数':             c.get('liked_count', 0),
                '子评论数':           c.get('sub_comment_count', 0),
                'IP地址':             c.get('ip_location',''),
                '一级评论ID':         (c.get('root_comment') or {}).get('id',''),
                '一级评论内容':       (c.get('root_comment') or {}).get('content',''),
                '引用的评论ID':       (c.get('quoted_comment') or {}).get('id',''),
                '引用的评论内容':     (c.get('quoted_comment') or {}).get('content',''),
            }
            all_comments.append(row)

        if not cursor:
            break

        time.sleep(pause)

    return all_comments

def main():
    NOTE_LINKS = [
        'https://www.xiaohongshu.com/explore/67fafda5000000001c03276e?xsec_token=ABzBY-JFjuu5wgUXe7ubk3CZKf4Oxk4sfswzREtsXaGdM=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f71d6f000000001d007966?xsec_token=AB_CSQ2YRNG5VxBV0ykHV2_oUZMY4Y9T4Af3AcZaWbxsc=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f93881000000001201f3a5?xsec_token=ABsXI1jccxTvHgGN-gDkjtTynN-ko8VMaZ9BXaCc_NRKQ=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f6796e000000001c00ee18?xsec_token=ABqvwn9IJ7cbTlWVLMCmmjR__IEyoyobv-vSwIzBn9Ad0=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f66bdc000000001c031006?xsec_token=ABKqN9mLi6gUfG-wVssXkOVwi6Zy7bumT7uaj_Oug08pU=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f72597000000001d02ed24?xsec_token=ABol5JHfVR_G7luJCeI47-FjrXz259D2QvUAmZhz6VBX4=&xsec_source=pc_search&source=unknown',
        'https://www.xiaohongshu.com/explore/67f7cc8c000000001d020f39?xsec_token=AB_RpTtD3rorml_noKZDchuAMkCMFyg1NDp0tlAUH6Ti8=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f5cf2e000000001a0042cd?xsec_token=ABgMlgY2dIoPPwNQhI2-yaZW5wX5k6Suye9VeIib7u9k0=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f874d2000000001d00a78b?xsec_token=ABuplbA2uWrd_6kcK1raLMcmLaJqx-sWr2udrha1pdASI=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67d6cba8000000001b03ee83?xsec_token=ABQPZOevIjyu7cv8vIqC-FGWfP0FDK4Cb8HTU1ef_DHIc=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67eeab00000000000b015dab?xsec_token=ABRnyOAM4fEodnQWKk9ZCvTFFcWAEVUD_cUNFuN77XTgM=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67efeb25000000001c016fb1?xsec_token=ABbahyNodkyHOvCGNpzd3VPQthyDd88hkwazpMQXIohxU=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67fa4439000000000b0150f3?xsec_token=ABtfVP7nq9mxp37lHl-zxRBCUTOgcVTlGZ1Eovzhp5vr0=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67fab8f4000000001c00b885?xsec_token=ABMvdmWzcz2eXeW2FyUBGlfJjSw1qloGVIDgQsgc4heVo=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67fb7f14000000000f032600?xsec_token=AB1utiinq7GQ_p27JsejJLDeaurj-P4bCL6yuaiuPVGQ8=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f0c46e000000000b02c6c8?xsec_token=ABZSkBcS3X1XYStE5R36IMXVgkdqPRPJOCXeVwXX4C_gw=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f80f3a000000001d01c9ae?xsec_token=ABq9y4KJp_TIBXbYTIXVrUMOgkx1CN4IXpwGd7MfRlGC4=&xsec_source=pc_search&source=unknown',
        'https://www.xiaohongshu.com/explore/67f7d8db000000001d0235db?xsec_token=AB_CSQ2YRNG5VxBV0ykHV2_mS7yef83ckvv1xgog4Vn8A=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/68011b2e000000001d002de9?xsec_token=ABor0Zo-giSm25pfYiNolMeoNGNwDofuspjKEBWUbr4mk=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f669a8000000001c03daed?xsec_token=ABKqN9mLi6gUfG-wVssXkOVw0lGrYj-NAo3E6JituPPWU=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f654690000000009015b4f?xsec_token=ABrFGFb3VYcv4oLraOI0vNUqm83l9RWOoKeMQP0BnRHRY=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f70eb9000000000900f185?xsec_token=AB_CSQ2YRNG5VxBV0ykHV2_lpcdQYRkIoqVA_a_qh9WAw=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f72e0a000000001d00a09d?xsec_token=AB_RpTtD3rorml_noKZDchuJFESFw9fyGTFJ5D3CXaX0s=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/6807a66e000000000e0059b8?xsec_token=AB2KsP43tO4aZTtxC2aHIlXQSYxkttAiPA5SLj6FBmo-8=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f37b66000000000f031e22?xsec_token=ABIFj8wlVgnCnOR8EEzgnXNBClJt5EYHJWDa1Ldc0yAM4=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67ff7b64000000001b039ebb?xsec_token=ABNqXf4d4pJFell0h_rUuQkAwo5O_WKrd68IIRJ2UPT1k=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67fb61d1000000001200f1ea?xsec_token=ABwRzLomL8Gfmx2ZS-dh6L_nmlbCgwGISRXi9CScKoNRY=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67fd03b7000000001e006fbf?xsec_token=ABPkenh9ggqGeDkQovPHQqyLA2qwqgOdrxyckWUKmWeH8=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f763bf000000000b02fa59?xsec_token=AB_CSQ2YRNG5VxBV0ykHV2_lWau2M1gWjHANtHYkcMVrc=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/6739f386000000001a037d87?xsec_token=ABAf4c4Vxch6RcQCY10tVXHsxpDY95Umjn9ntC-tFWIvI=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f0c46e000000000b02c6c8?xsec_token=ABZSkBcS3X1XYStE5R36IMXTV5_-wKkU0wm8IjLVOue7o=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f60ce5000000000b01ff39?xsec_token=ABKqN9mLi6gUfG-wVssXkOV13FpdK-SVb3r-PFsaWDc2o=&xsec_source=pc_search&source=web_search_result_notes',
        'https://www.xiaohongshu.com/explore/67f662e0000000000f0388ee?xsec_token=ABKqN9mLi6gUfG-wVssXkOVweXmqJnFVt1JE6ucuj3qJU=&xsec_source=pc_search&source=web_explore_feed',
        'https://www.xiaohongshu.com/explore/67f669a8000000001c03daed?xsec_token=ABKqN9mLi6gUfG-wVssXkOV3ODMOtx6P06nZltJdyMEJ0=&xsec_source=pc_search&source=web_search_result_notes',
    ]

    headers = {
        "User-Agent": "Xhs-Android/8.41.0 (Android;10)",
        "X-T": "<请填入抓包获得的 X-T>",
        "X-Sign": "<请填入抓包获得的 X-Sign>",
        "Cookie": "<请填入你的 Cookie>"
    }

    all_data = []
    for link in NOTE_LINKS:
        print(f"正在处理: {link}")
        rows = fetch_comments(link, headers)
        print(f"  抓取到 {len(rows)} 条评论\n")
        all_data.extend(rows)

    if all_data:
        df = pd.DataFrame(all_data)
        # 用 utf-8-sig，让 Excel 直接双击就能正常识别中文
        df.to_csv('xiaohongshu_comments_by_link.csv', index=False, encoding='utf-8-sig')
        print("结果已保存至 xiaohongshu_comments_by_link.csv")
    else:
        print("未获取到任何评论，请检查链接或 Headers 是否正确。")

if __name__ == "__main__":
    main()