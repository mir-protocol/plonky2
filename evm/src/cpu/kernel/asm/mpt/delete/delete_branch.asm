// Delete from a branch node.
// Assumes that `num_nibbles>0` and that the value of the branch node is zero.
// TODO: Depending on how we implement the receipt trie, may need to revisit these assumptions.
global mpt_delete_branch:
    // stack: node_type, node_payload_ptr, num_nibbles, key, retdest
    POP
    // stack: node_payload_ptr, num_nibbles, key, retdest
    DUP2 ISZERO %jumpi(panic) // This should never happen.
    DUP3 DUP3
    // stack: num_nibbles, key, node_payload_ptr, num_nibbles, key, retdest
    %split_first_nibble
    %stack (first_nibble, num_nibbles, key, node_payload_ptr, old_num_nibbles, old_key) ->
        (node_payload_ptr, first_nibble, num_nibbles, key, after_mpt_delete_branch, first_nibble, node_payload_ptr)
    ADD
    // stack: child_ptr_ptr, num_nibbles, key, after_mpt_delete_branch, first_nibble, node_payload_ptr, retdest
    %mload_trie_data
    %jump(mpt_delete)

after_mpt_delete_branch:
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP1 %mload_trie_data ISZERO %jumpi(maybe_normalize_branch)
update_branch:
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %get_trie_data_size
    // stack: updated_branch_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    PUSH @MPT_NODE_BRANCH %append_to_trie_data
    // stack: updated_branch_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP4                %mload_trie_data %append_to_trie_data // Copy child[0]
    DUP4 %add_const(1)  %mload_trie_data %append_to_trie_data // ...
    DUP4 %add_const(2)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(3)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(4)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(5)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(6)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(7)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(8)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(9)  %mload_trie_data %append_to_trie_data
    DUP4 %add_const(10) %mload_trie_data %append_to_trie_data
    DUP4 %add_const(11) %mload_trie_data %append_to_trie_data
    DUP4 %add_const(12) %mload_trie_data %append_to_trie_data
    DUP4 %add_const(13) %mload_trie_data %append_to_trie_data
    DUP4 %add_const(14) %mload_trie_data %append_to_trie_data
    DUP4 %add_const(15) %mload_trie_data %append_to_trie_data // Copy child[15]
    DUP4 %add_const(16) %mload_trie_data %append_to_trie_data // Copy value_ptr
    // stack: updated_branch_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    SWAP1
    // stack: updated_child_ptr, updated_branch_ptr, first_nibble, node_payload_ptr, retdest
    DUP2 %increment DUP4 ADD
    // stack: updated_branch_ptr+first_nibble, updated_child_ptr, updated_branch_ptr, first_nibble, node_payload_ptr, retdest
    %mstore_trie_data
    %stack (updated_branch_ptr, first_nibble, node_payload_ptr, retdest) -> (retdest, updated_branch_ptr)
    JUMP

maybe_normalize_branch:
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    PUSH 0 %mstore_kernel_general(0) PUSH 0 %mstore_kernel_general(1)
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    PUSH 0
loop:
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP1 DUP4 EQ %jumpi(loop_eq_first_nibble)
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP1 %eq_const(16) %jumpi(loop_end)
    DUP1 DUP5 ADD %mload_trie_data %mload_trie_data ISZERO ISZERO %jumpi(loop_non_empty)
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %increment %jump(loop)
loop_eq_first_nibble:
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %increment %jump(loop)
loop_non_empty:
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %mload_kernel_general(0) %increment %mstore_kernel_general(0)
    DUP1 %mstore_kernel_general(1)
    %increment %jump(loop)
loop_end:
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    POP
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %mload_kernel_general(0) %gt_const(1) %jumpi(update_branch)
    %mload_kernel_general(0) ISZERO %jumpi(panic) // TODO: Remove. For debugging only.
    // stack: updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %mload_kernel_general(1)
    // stack: i, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP4 ADD %mload_trie_data
    // stack: only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP1 %mload_trie_data %eq_const(@MPT_NODE_BRANCH)     %jumpi(maybe_normalize_branch_branch)
    DUP1 %mload_trie_data %eq_const(@MPT_NODE_EXTENSION)  %jumpi(maybe_normalize_branch_leafext)
    DUP1 %mload_trie_data %eq_const(@MPT_NODE_LEAF)       %jumpi(maybe_normalize_branch_leafext)
    PANIC // This should never happen.

maybe_normalize_branch_branch:
    // stack: only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %get_trie_data_size // pointer to the extension node we're about to create
    // stack: extension_ptr, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    PUSH @MPT_NODE_EXTENSION %append_to_trie_data
    // stack: extension_ptr, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    PUSH 1 %append_to_trie_data // Append node_len to our node
    // stack: extension_ptr, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    %mload_kernel_general(1) %append_to_trie_data // Append node_key to our node
    // stack: extension_ptr, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    SWAP1 %append_to_trie_data // Append updated_child_node_ptr to our node
    %stack (extension_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest) -> (retdest, extension_ptr)
    JUMP

maybe_normalize_branch_leafext:
    // stack: only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP1 %increment %mload_trie_data
    // stack: child_len, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    DUP2 %add_const(2) %mload_trie_data
    // stack: child_key, child_len, only_child_ptr, updated_child_ptr, first_nibble, node_payload_ptr, retdest
    SWAP2 %add_const(3) %mload_trie_data
    %mload_kernel_general(1)
    %stack (i, child_value_ptr, child_len, child_key, updated_child_ptr, first_nibble, node_payload_ptr) ->
        (1, i, child_len, child_key, child_value_ptr)
    %merge_nibbles
    // stack: len, key, value_ptr, retdest
    %get_trie_data_size // pointer to the leaf node we're about to create
    // stack: leaf_ptr, len, key, value_ptr, retdest
    PUSH @MPT_NODE_LEAF %append_to_trie_data
    // stack: leaf_ptr, len, key, value_ptr, retdest
    SWAP1 %append_to_trie_data // Append len to our node
    // stack: leaf_ptr, key, value_ptr, retdest
    SWAP1 %append_to_trie_data // Append key to our node
    // stack: leaf_ptr, value_ptr, retdest
    SWAP1 %append_to_trie_data // Append value_ptr to our node
    // stack: leaf_ptr, retdest
    SWAP1 JUMP
