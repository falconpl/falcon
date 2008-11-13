	.file	"dynlib_sys_gcc.cpp"
	.text
	.align 2
.globl _ZN6Falcon3Sys16dynlib_void_callEPvPhj
	.type	_ZN6Falcon3Sys16dynlib_void_callEPvPhj, @function
_ZN6Falcon3Sys16dynlib_void_callEPvPhj:
.LFB163:
	pushl	%ebp
.LCFI0:
	movl	%esp, %ebp
.LCFI1:
	pushl	%esi
.LCFI2:
	movl	16(%ebp), %eax
	shrl	$2, %eax
	movl	%eax, 16(%ebp)
	movl	8(%ebp), %edx
	movl	12(%ebp), %esi
	movl	16(%ebp), %ecx
#APP
	1: orl   %ecx, %ecx
jz    2f
movl  (%esi),%eax
pushl %eax
addl  $4,%esi
decl  %ecx
jmp   1b
2: call  *%edx
movl  %ebp, %esp
popl  %ebp
ret

#NO_APP
	popl	%esi
	popl	%ebp
	ret
.LFE163:
	.size	_ZN6Falcon3Sys16dynlib_void_callEPvPhj, .-_ZN6Falcon3Sys16dynlib_void_callEPvPhj
.globl __gxx_personality_v0
	.align 2
.globl _ZN6Falcon3Sys17dynlib_voidp_callEPvPhj
	.type	_ZN6Falcon3Sys17dynlib_voidp_callEPvPhj, @function
_ZN6Falcon3Sys17dynlib_voidp_callEPvPhj:
.LFB164:
	pushl	%ebp
.LCFI3:
	movl	%esp, %ebp
.LCFI4:
	pushl	%esi
.LCFI5:
	movl	16(%ebp), %eax
	shrl	$2, %eax
	movl	%eax, 16(%ebp)
	movl	8(%ebp), %edx
	movl	12(%ebp), %esi
	movl	16(%ebp), %ecx
#APP
	1: orl   %ecx, %ecx
jz    2f
movl  (%esi),%eax
pushl %eax
addl  $4,%esi
decl  %ecx
jmp   1b
2: call  *%edx
movl  %ebp, %esp
popl  %ebp
ret

#NO_APP
	movl	$0, %eax
	popl	%esi
	popl	%ebp
	ret
.LFE164:
	.size	_ZN6Falcon3Sys17dynlib_voidp_callEPvPhj, .-_ZN6Falcon3Sys17dynlib_voidp_callEPvPhj
	.align 2
.globl _ZN6Falcon3Sys17dynlib_dword_callEPvPhj
	.type	_ZN6Falcon3Sys17dynlib_dword_callEPvPhj, @function
_ZN6Falcon3Sys17dynlib_dword_callEPvPhj:
.LFB165:
	pushl	%ebp
.LCFI6:
	movl	%esp, %ebp
.LCFI7:
	pushl	%esi
.LCFI8:
	movl	16(%ebp), %eax
	shrl	$2, %eax
	movl	%eax, 16(%ebp)
	movl	8(%ebp), %edx
	movl	12(%ebp), %esi
	movl	16(%ebp), %ecx
#APP
	1: orl   %ecx, %ecx
jz    2f
movl  (%esi),%eax
pushl %eax
addl  $4,%esi
decl  %ecx
jmp   1b
2: call  *%edx
movl  %ebp, %esp
popl  %ebp
ret

#NO_APP
	movl	$0, %eax
	popl	%esi
	popl	%ebp
	ret
.LFE165:
	.size	_ZN6Falcon3Sys17dynlib_dword_callEPvPhj, .-_ZN6Falcon3Sys17dynlib_dword_callEPvPhj
	.align 2
.globl _ZN6Falcon3Sys17dynlib_qword_callEPvPhj
	.type	_ZN6Falcon3Sys17dynlib_qword_callEPvPhj, @function
_ZN6Falcon3Sys17dynlib_qword_callEPvPhj:
.LFB166:
	pushl	%ebp
.LCFI9:
	movl	%esp, %ebp
.LCFI10:
	pushl	%esi
.LCFI11:
	movl	16(%ebp), %eax
	shrl	$2, %eax
	movl	%eax, 16(%ebp)
	movl	8(%ebp), %edx
	movl	12(%ebp), %esi
	movl	16(%ebp), %ecx
#APP
	1: orl   %ecx, %ecx
jz    2f
movl  (%esi),%eax
pushl %eax
addl  $4,%esi
decl  %ecx
jmp   1b
2: call  *%edx
movl  %ebp, %esp
popl  %ebp
ret

#NO_APP
	movl	$0, %eax
	movl	$0, %edx
	popl	%esi
	popl	%ebp
	ret
.LFE166:
	.size	_ZN6Falcon3Sys17dynlib_qword_callEPvPhj, .-_ZN6Falcon3Sys17dynlib_qword_callEPvPhj
	.align 2
.globl _ZN6Falcon3Sys18dynlib_double_callEPvPhj
	.type	_ZN6Falcon3Sys18dynlib_double_callEPvPhj, @function
_ZN6Falcon3Sys18dynlib_double_callEPvPhj:
.LFB167:
	pushl	%ebp
.LCFI12:
	movl	%esp, %ebp
.LCFI13:
	pushl	%esi
.LCFI14:
	movl	16(%ebp), %eax
	shrl	$2, %eax
	movl	%eax, 16(%ebp)
	movl	8(%ebp), %edx
	movl	12(%ebp), %esi
	movl	16(%ebp), %ecx
#APP
	1: orl   %ecx, %ecx
jz    2f
movl  (%esi),%eax
pushl %eax
addl  $4,%esi
decl  %ecx
jmp   1b
2: call  *%edx
movl  %ebp, %esp
popl  %ebp
ret

#NO_APP
	fldz
	popl	%esi
	popl	%ebp
	ret
.LFE167:
	.size	_ZN6Falcon3Sys18dynlib_double_callEPvPhj, .-_ZN6Falcon3Sys18dynlib_double_callEPvPhj
	.ident	"GCC: (GNU) 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)"
	.section	.note.GNU-stack,"",@progbits
