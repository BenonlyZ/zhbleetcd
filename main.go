package main

import (
	"sort"
	"unicode/utf8"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func main() {

}

//判断括号字符串是否有效
func isValid(s string) bool {
	var str = make([]byte, 0)
	count := len(str)
	for i := 0; i < len(s); i++ {

		if s[i] == '(' || s[i] == '[' || s[i] == '{' {
			str = append(str, s[i])
			count++
		} else {
			if count == 0 {
				return false
			}
			switch s[i] {
			case ')':
				if str[count-1] == '(' {
					str = str[0 : count-1]
					count--
				} else {
					return false
				}
			case ']':
				if str[count-1] == '[' {
					str = str[0 : count-1]
					count--
				} else {
					return false
				}
			case '}':
				if str[count-1] == '{' {
					str = str[0 : count-1]
					count--
				} else {
					return false
				}
			}
		}
	}
	if count == 0 {
		return true
	} else {
		return false
	}
}

//整数反转
func reverse(x int) int {
	if x == 0 {
		return 0
	}

	res := 0
	for x != 0 {
		y := x % 10
		x = x / 10
		res = res*10 + y
	}

	n := 1 << 31
	if res < (-n) || res > n-1 {
		return 0
	} else {
		return res
	}
}

//是否为回文数
func isPalindrome(x int) bool {
	tmp := x
	n := 1 << 31
	if tmp < 0 || tmp > n-1 {
		return false
	}
	var y int
	var res int
	for x != 0 {
		y = x % 10
		x = x / 10
		res = res*10 + y
	}

	if res == tmp {
		return true
	} else {
		return false
	}
}

//搜索插入位置
func searchInsert(nums []int, target int) int {
	high := len(nums) - 1
	low := 0
	var mid int
	for low <= high {
		mid = (low + high) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}

	if target > nums[mid] {
		return mid + 1
	} else {
		return mid
	}
}

//罗马数字转整数
func romanToInt(s string) int {
	var count int
	for i := len(s) - 1; i >= 0; i-- {
		switch s[i] {
		case 'I':
			count = count + 1
		case 'V':
			if i-1 >= 0 && s[i-1] == 'I' {
				count = count + 4
				i--
			} else {
				count = count + 5
			}
		case 'X':
			if i-1 >= 0 && s[i-1] == 'I' {
				count = count + 9
				i--
			} else {
				count = count + 10
			}

		case 'L':
			if i-1 >= 0 && s[i-1] == 'X' {
				count = count + 40
				i--
			} else {
				count = count + 50
			}
		case 'C':
			if i-1 >= 0 && s[i-1] == 'X' {
				count = count + 90
				i--
			} else {
				count = count + 100
			}
		case 'D':
			if i-1 >= 0 && s[i-1] == 'C' {
				count = count + 400
				i--
			} else {
				count = count + 500
			}
		case 'M':
			if i-1 >= 0 && s[i-1] == 'C' {
				count = count + 900
				i--
			} else {
				count = count + 1000
			}
		}
	}

	return count
}

//最后一个单词长度，注意最后为'a '也算一个单词，长度为1
func lengthOfLastWord(s string) int {
	var count int
	l := len(s) - 1

	for l >= 0 && s[l] == ' ' {
		l--
	}
	for i := l; i >= 0; i-- {
		if s[i] == ' ' {
			break
		}
		count++
	}
	return count
}

//链表两数相加
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var tmpl = new(ListNode)
	sl := tmpl
	for l1 != nil || l2 != nil {
		var nextnode = new(ListNode)
		switch {
		case l1 != nil && l2 != nil:
			p := tmpl.Val + l1.Val + l2.Val
			if p >= 10 {
				tmpl.Val = p - 10
				nextnode.Val += 1
			} else {
				tmpl.Val = p
			}
			l1 = l1.Next
			l2 = l2.Next
			if l1 == nil && l2 == nil && nextnode.Val == 0 {
				break
			}
			tmpl.Next = nextnode
			tmpl = tmpl.Next

		case l1 != nil:
			p := tmpl.Val + l1.Val
			if p >= 10 {
				tmpl.Val = p - 10
				nextnode.Val += 1
			} else {
				tmpl.Val = p
			}
			l1 = l1.Next
			if l1 == nil && nextnode.Val == 0 {
				break
			}
			tmpl.Next = nextnode
			tmpl = tmpl.Next

		case l2 != nil:
			p := tmpl.Val + l2.Val
			if p >= 10 {
				tmpl.Val = p - 10
				nextnode.Val += 1
			} else {
				tmpl.Val = p

			}
			l2 = l2.Next
			if l2 == nil && nextnode.Val == 0 {
				break
			}
			tmpl.Next = nextnode
			tmpl = tmpl.Next

		}
	}
	return sl
}

//字符串最长公共子前缀
func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	minlenindex := 0
	for p := 1; p < len(strs); p++ {
		if len(strs[p]) < len(strs[minlenindex]) {
			minlenindex = p
		}
	}
	var count int
	for i := 0; i < len(strs[minlenindex]); i++ {

		var j int
		for j = 0; j < len(strs); j++ {
			if j == minlenindex {
				continue
			}
			if strs[minlenindex][i] != strs[j][i] {
				break
			}
		}
		if j != len(strs) {
			break
		}
		count++
	}
	return strs[minlenindex][0:count]
}

//计算数组中只出现一次的元素,异或运算
func singleNumber(nums []int) int {
	count := nums[0]
	for i := 1; i < len(nums); i++ {
		count ^= nums[i]
	}
	return count
}

//二进制求和
func addBinary(a string, b string) string {
	al := len(a)
	bl := len(b)
	if al > bl {
		tmp := al - bl
		var str string
		for i := 0; i < tmp; i++ {
			str += "0"
		}
		b = str + b
	} else {
		tmp := bl - al
		var str string
		for i := 0; i < tmp; i++ {
			str += "0"
		}
		a = str + a
	}
	var n = make([]byte, len(a)+1)
	for m := 0; m < len(a)+1; m++ {
		n[m] = '0'
	}
	for j := len(a) - 1; j >= 0; j-- {
		switch int(a[j]) + int(b[j]) + int(n[j+1]) {
		case 144:
			n[j+1] = '0'
		case 145:
			n[j+1] = '1'
		case 146:
			n[j+1] = '0'
			n[j] = '1'
		case 147:
			n[j+1] = '1'
			n[j] = '1'
		}
	}
	if n[0] == '0' {
		return string(n[1:])
	} else {
		return string(n)
	}
}

//求数组里的多数元素，用摩尔投票法，则不同数之间相互抵消，剩下的元素则为数量超过n/2的元素
func majorityElement(nums []int) int {
	count := 1
	maj := nums[0]
	for i := 1; i < len(nums)-1; i++ {
		if maj == nums[i] {
			count++
		} else {
			count--
			if count == 0 {
				maj = nums[i+1]

			}
		}
	}
	return maj
}

//爬楼梯(未通过，待解决)
func climbStairs(n int) int {
	m := make(map[int]int)
	for i := 0; i <= n; i++ {
		for j := 0; j <= n/2; j++ {
			if (i + 2*j) == n {
				m[i] = j
				break
			}
		}
	}
	var count int
	for p, q := range m {
		if p == 0 || q == 0 {
			count++
			continue
		}
		p1 := 1
		for i := 2; i <= p; i++ {
			p1 *= i
		}
		q1 := 1
		for j := 2; j <= q; j++ {
			q1 *= j
		}
		pq := 1
		for f := 2; f <= (p + q); f++ {
			pq *= f
		}
		count += ((pq / p1) / q1)
	}
	return count
}

//判断字符串是否为回文串
func isPalindrome2(s string) bool {
	low := 0
	high := len(s) - 1

	for low < len(s) && high >= 0 {
		// 48 57  65 90  97 122
		if !((s[low] >= 48 && s[low] <= 57) || (s[low] >= 65 && s[low] <= 90) || (s[low] >= 97 && s[low] <= 122)) {
			low++
			continue
		}
		if !((s[high] >= 48 && s[high] <= 57) || (s[high] >= 65 && s[high] <= 90) || (s[high] >= 97 && s[high] <= 122)) {
			high--
			continue
		}

		if s[low] == s[high] || (((s[low] >= 65 && s[low] <= 90) || (s[low] >= 97 && s[low] <= 122) && (s[high] >= 65 && s[high] <= 90) || (s[high] >= 97 && s[high] <= 122)) && (s[low]+32 == s[high] || s[low]-32 == s[high] || s[low] == s[high]+32 || s[low] == s[high]-32)) {
			low++
			high--
			if low >= high {
				return true
			}
		} else {
			return false
		}
	}
	return true
}

//检测链表是否带环,一般设置一快一慢两指针,slow每走一步，fast走两步(fast速度为slow二倍比较好);当初次同时入环时(循环链表),相遇节点也为入环节点;其他情况在环内相遇
func hasCycle(head *ListNode) bool {
	fast := head
	slow := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

//找到链表的入环口节点
func detectCycle(head *ListNode) *ListNode {
	fast := head
	slow := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
		if slow == fast {
			meet := fast
			for meet != head {
				meet = meet.Next
				head = head.Next
			}
			return meet
		}
	}
	return nil
}

//有效的字母异位词
func isAnagram(s string, t string) bool {
	smap := make(map[rune]int)
	for _, v := range s {
		smap[v] += 1
	}

	for _, u := range t {
		_, ok := smap[u]
		if ok {
			smap[u] -= 1
		} else {
			return false
		}
	}
	for _, w := range smap {
		if w != 0 {
			return false
		}
	}
	return true
}

//同构字符串
func isIsomorphic(s string, t string) bool {
	var smap = make(map[byte]byte)
	for i := 0; i < len(s); i++ {
		val, ok := smap[s[i]]
		if ok {
			if val != t[i] {
				return false
			}
		} else {
			smap[s[i]] = t[i]
		}
	}
	var tmap = make(map[byte]int)
	for _, v := range smap {
		_, ok := tmap[v]
		if ok {
			return false
		} else {
			tmap[v] = 1
		}
	}
	return true
}

//最大子序和
func maxSubArray(nums []int) int {
	//定义过去数之和的最大值
	max := nums[0]
	var count int
	for _, v := range nums {
		//如果过去数之和为正整数,则当前数直接加
		if count >= 0 {
			count += v
		} else {
			//如果过去数之和为负数,则重新累加,从当前数开始
			count = v
		}
		//如果经过处理后的过去数之和count大于max,则max重新附值count
		if count > max {
			max = count
		}
	}
	return max
}

//丢失的数字
func missingNumber(nums []int) int {
	smap := make(map[int]int)
	n := len(nums)
	for _, v := range nums {
		smap[v] = 1
	}

	for i := 0; i <= n; i++ {
		_, ok := smap[i]
		if !ok {
			return i
		}
	}
	return -1
}

//有序数组的平方
func sortedSquares(nums []int) []int {
	var i int
	ln := len(nums)
	for i = 0; i < ln; i++ {
		if nums[i] >= 0 {
			break
		}
	}
	switch i {
	case 0:
		for j := 0; j < ln; j++ {
			nums[j] = nums[j] * nums[j]
		}
		return nums
	case ln:
		smun := make([]int, ln)
		for j := 0; j < ln; j++ {
			smun[ln-1-j] = nums[j] * nums[j]
		}
		return smun
	default:
		num1 := make([]int, i)
		num2 := make([]int, ln-i)
		for j := 0; j < i; j++ {
			num1[i-1-j] = nums[j] * nums[j]
		}
		for j := 0; j < ln-i; j++ {
			num2[j] = nums[i+j] * nums[i+j]
		}
		low1, low2 := 0, 0
		nums = nil
		for low1 < len(num1) && low2 < len(num2) {
			if num1[low1] < num2[low2] {
				nums = append(nums, num1[low1])
				low1++
			} else {
				nums = append(nums, num2[low2])
				low2++
			}
		}
		if low1 < len(num1) {
			nums = append(nums, num1[low1:]...)
		}
		if low2 < len(num2) {
			nums = append(nums, num2[low2:]...)
			low2++
		}
		return nums
	}
}

//三数之和, 思路：先排序，然后对数组for循环。每层循环以当前元素为n1，n2为n1后面一位的元素，n3为最末尾的元素。
//判断：如果n1+n2+n3<0,则需要增大，n2后移一位;如果>0，则需要缩小,n3前移一位；当=0时，则n1,n2,n3为一组。继续知道n2=n3为止
//继续for循环，对nums[1],nums[2]...做相同操作
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	res := make([][]int, 0)

	for i := 0; i < len(nums)-2; i++ {
		n1 := nums[i]
		//当n1大于0，则后面的数都大于0，则一定不存在相加=0的一组元素
		if n1 > 0 {
			break
		}
		//由于不能出现重复，需要去重
		if i > 0 && n1 == nums[i-1] {
			continue
		}
		l, r := i+1, len(nums)-1
		for l < r {
			n2, n3 := nums[l], nums[r]
			if n1+n2+n3 == 0 {
				res = append(res, []int{n1, n2, n3})
				//也是去重操作
				for l < r && nums[l] == n2 {
					l++
				}
				for l < r && nums[r] == n3 {
					r--
				}

			} else if n1+n2+n3 < 0 {
				l++
			} else {
				r--
			}
		}
	}
	return res
}

//位运算本身是类似电路题，其中异或运算用处最大，异或运算主要是三个性质：
//(1)数字自身异或为0；(2)任何数字与0异或不变；(3)异或运算满足交换律。——可用于抵消偶次数出现的数字等
//两字符串s,t;t比s多添加一个字母，其余相同，请找出在 t 中被添加的字母。
func findTheDifference(s string, t string) byte {
	var n uint8
	var i int
	for i = 0; i < utf8.RuneCountInString(s); i++ {
		n ^= s[i]
		n ^= t[i]
	}
	n ^= t[i]
	return n
}
