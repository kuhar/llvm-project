	.section	.AMDGPU.config,"",@progbits
	.long	47176
	.long	0
	.long	47180
	.long	0
	.long	47200
	.long	0
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	"sramecc-subtarget-feature-enabled" ; -- Begin function sramecc-subtarget-feature-enabled
	.p2align	2
	.type	"sramecc-subtarget-feature-enabled",@function
"sramecc-subtarget-feature-enabled":    ; @sramecc-subtarget-feature-enabled
; %bb.0:
	s_wait_loadcnt_dscnt 0x0
	s_wait_kmcnt 0x0
	s_set_pc_i64 s[30:31]
.Lfunc_end0:
	.size	"sramecc-subtarget-feature-enabled", .Lfunc_end0-"sramecc-subtarget-feature-enabled"
                                        ; -- End function
	.set "sramecc-subtarget-feature-enabled.num_vgpr", 0
	.set "sramecc-subtarget-feature-enabled.num_agpr", 0
	.set "sramecc-subtarget-feature-enabled.numbered_sgpr", 32
	.set "sramecc-subtarget-feature-enabled.num_named_barrier", 0
	.set "sramecc-subtarget-feature-enabled.private_seg_size", 0
	.set "sramecc-subtarget-feature-enabled.uses_vcc", 0
	.set "sramecc-subtarget-feature-enabled.uses_flat_scratch", 0
	.set "sramecc-subtarget-feature-enabled.has_dyn_sized_stack", 0
	.set "sramecc-subtarget-feature-enabled.has_recursion", 0
	.set "sramecc-subtarget-feature-enabled.has_indirect_call", 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 12
; TotalNumSgprs: 32
; NumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.section	.AMDGPU.csdata,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.amd_amdgpu_isa "amdgcn----gfx1250"
