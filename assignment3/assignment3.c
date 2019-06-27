/*
 ============================================================================
 Filename    : assignment3.c
 Author      : Dominique Roduit (234868)
 Date        : Nov. 13th, 2017
 ============================================================================
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node *next;
    omp_lock_t lock;
} node_t;


/*
 * This function initializes a new linked list.
 */
void init_list(node_t **head, int val) {
    (*head) = malloc(sizeof(node_t));
    
    (*head)->val = val;
    (*head)->next = NULL;
    
    // We make the freshly created node able to hold a lock
    omp_init_lock(&((*head)->lock));
}

/*
 * This function prints all the elements of a given linked list.
 */
void print_list(node_t *head) {
	node_t* current = head;
    node_t* previous = NULL;
    
    while (current != NULL) {
		 // We lock the current node as we traverse the list
		omp_set_lock(&current->lock);
		
        printf("%d\n", current->val);  
		     
		previous = current;
        current = current->next;
        
        // We release the lock on the current node after current->next
		omp_unset_lock(&previous->lock);
    }
    
}

/*
 * This function counts the elements of a given linked list and returns the counted number.
 */
int count(node_t *head) {
    node_t *current = head;
    int count = 0;

	node_t *previous = NULL;

    while (current != NULL) {	
		// We lock the current node before to increment the counter.
		// That way, we avoid the case where another thread delete a
		// node or add a node in the same time, that would finally give
		// a wrong result for the counter.
		omp_set_lock(&current->lock);
		
		count++;		
		
		previous = current;
        current = current->next;
        
        omp_unset_lock(&previous->lock);
    }
    
    return count;
}

/*
 * This function appends a new given element to the end of a given linked list.
 */
void append(node_t *head, int val) {
    node_t *current = head;
    node_t *previous = NULL;
		
    while (current != NULL) {		
		// We lock the current node ...
		omp_set_lock(&current->lock);
		
		previous = current;
        current = current->next;
        
        // ... and release it 
		omp_unset_lock(&previous->lock);
    }
    
    current = previous;
    
    // We lock the last node of the list
    omp_set_lock(&current->lock);
    
    current->next = malloc(sizeof(node_t));
    current->next->val = val;
    current->next->next = NULL;
    // We init the lock of the new node
    omp_init_lock(&current->next->lock);
    
    // Value added successfully, we release the lock hold by
    // the predecessor of the newly added node
    omp_unset_lock(&current->lock);
}

/*
 * This function adds a new given element to the beginning of a given linked list.
 */
void add_first(node_t **head, int val) {
	if(*head == NULL)
		return;
		
	// We lock the current head which will then be the second node of the list after the new insertion.
	// We have to protect it, otherwise, for instance, another thread could delete it, 
	// and at the moment to make new_node->next = *head, we would have inconsistency
	node_t *old_head = *head;
	omp_set_lock(&old_head->lock);
	
	node_t *new_node;
	new_node = malloc(sizeof(node_t));
	new_node->val = val;
	new_node->next = *head;
	// We make the freshly created node able to hold a lock
	omp_init_lock(&new_node->lock);
	
	*head = new_node;
	
	// New head successfully added, we can release the old head
	omp_unset_lock(&old_head->lock);

}

/*
 * This function inserts a new given element at the specified position of a given linked list.
 * It returns 0 on a successful insertion, and -1 if the list is not long enough.
 */
int insert(node_t **head, int val, int index) {
    if (index == 0) {
        add_first(head, val);
        return 0;
    }
    
    node_t *current = *head;
    node_t *previous = NULL;
    
    if(current == NULL)
		return -1;
    
    for (int i = 0; i < index-1; i++) {
		// We lock the current node and progessively lock each node one after
		// the other until we reach the desired index.
		omp_set_lock(&current->lock);
		
        if (current->next == NULL) {
			// The list is not long enough to contain the node we want to add, 
			// then we release the lock on the node holding it before to return.
            omp_unset_lock(&current->lock);
            return -1;
        }
		
		previous = current;
        current = current->next;
        
        // ... and give the lock to the next node
        omp_unset_lock(&previous->lock);
    }
    
    // At this point of the code, "current" is the node standing just
    // before the new node we want to add, we lock it.
    omp_set_lock(&current->lock); 
    
    node_t *new_node;
    new_node = malloc(sizeof(node_t));
    new_node->val = val;
    new_node->next = current->next;
    // We make the freshly created node able to hold a lock
    omp_init_lock(&new_node->lock);
    
    current->next = new_node;
    
    // The new node was successfully added, the pointer of the node
    // before the newly added node is changed, and the pointer on the
    // next node was added correctly because the node containing the
    // pointer was locked. We can then release the lock on this node.
    omp_unset_lock(&current->lock);
    
    return 0;
}

/*
 * This function pops the first element of a given linked list.
 * The value of that element is returned (if list is not empty), and the element is removed.
 */
int pop(node_t **head) {
    int retval = -1;
    node_t* next_node = NULL;
    node_t* prev_node = NULL;
	
    if (*head == NULL) {
        return -1;
    }
    
    // We lock the head of the list (the node we will remove) ...
    omp_set_lock(&(*head)->lock);
	prev_node = (*head);
    next_node = (*head)->next;
    		
    retval = (*head)->val;
	*head = next_node;
    
    // We release and detroy the lock of the deleted node before to free its allocated memory
    omp_unset_lock(&prev_node->lock);   
	omp_destroy_lock(&prev_node->lock);

	free(prev_node);
	
    return retval;
}

/*
 * This function removes the specified element of a given linked list.
 * The value of that element is returned if the list is long enough; otherwise it returns -1.
 * Please note the index starts from 0.
 */
int remove_by_index(node_t **head, int index) {
    if (index == 0) {
        return pop(head);
    } 

    int retval = -1;
    node_t * current = *head;
    node_t * previous = NULL;
	
	if(current == NULL) {
		return -1;
	}

    for (int i = 0; i <= index-1; i++) {
		// We  will progressively give the lock to the next node 
		// as we traverse the list, until we reach the desired index.
		omp_set_lock(&current->lock);
	
        if (current->next == NULL) {
            // The list is not long enough to contain the node we want to remove at the given index, 
			// then we release the lock on the node holding it before to return.
            omp_unset_lock(&current->lock);
            return -1;
        }
        
        previous = current;
        current = current->next;
        
        // The next node exists, so we can release the lock on the current node ...
        omp_unset_lock(&previous->lock);
    }
    
    // At this point of the code, "previous" is the node standing just
    // before the new node we want to remove, and "current" is the node to remove.
    // We lock both of them
	omp_set_lock(&previous->lock);
    omp_set_lock(&current->lock);

    // Current node has to be removed!
    previous->next = current->next;
    retval = current->val;
    
    omp_unset_lock(&current->lock);
    // We detroy the lock of the node before to delete it (by freeing its allocated memory)
    omp_destroy_lock(&current->lock);
    
    free(current);
    current = NULL;
    
    // The node at the specified indec was successfully removed.
    // We can release the lock on the previous node.
    omp_unset_lock(&previous->lock);

    return retval;
}

/*
 * This function removes the specified element of a given linked list.
 * The value of that element is returned if the element is found; otherwise it returns -1.
 */
int remove_by_value(node_t **head, int val) {
	
    if (*head == NULL) {
        return -1;
    }
     
    if ((*head)->val == val) {
        return pop(head);
    }
    
    // We start by locking the head of the list
    omp_set_lock(&(*head)->lock);
    
    // If the list only contains the head and it doesn't contain the value wanted
    if((*head)->next == NULL) {
		// We unlock the head and return that the value doesn't exist
		omp_unset_lock(&(*head)->lock);
		return -1;
	}
    
    node_t* previous = *head;
    node_t* current = (*head)->next;
    
    omp_set_lock(&current->lock);
    
    while (current->next != NULL) {
	
        if (current->val == val) {
            previous->next = current->next;
           
            // The node containing the value to remove was found. Release the lock ...
            omp_unset_lock(&current->lock);
            // ... and destroy it before to free the node's allocated memory
            omp_destroy_lock(&current->lock);
           
            free(current);
            current = NULL;
            
            // release the lock on the node standing before the deleted node 
            omp_unset_lock(&previous->lock);
            return val;
        }
			
		omp_unset_lock(&previous->lock);
        
        previous = current;
        current  = current->next;
		
		// Give the lock to the next node
		// The value is not in this node, release the lock on the previous node
        omp_set_lock(&current->lock);
    }
    
    // We check if the last node has the value wanted
    if(current->val == val) {
		previous->next = current->next;
		omp_unset_lock(&current->lock);
		omp_destroy_lock(&current->lock);
		free(current);
		current = NULL;
		omp_unset_lock(&previous->lock);
		return val;
	}
    
    // we didn't find any node containing val in the list 
	omp_unset_lock(&current->lock);
	omp_unset_lock(&previous->lock);
  
    return -1;
}

/*
 * This function searched for a specified element in a given linked list.
 * The index of that element is returned if the element is found; otherwise it returns -1.
 */
int search_by_value(node_t *head, int val) {
    node_t *current = head;
    node_t *previous = NULL;
    
    int index = 0;
    
    if (current == NULL) {
        return -1;
    }
    
    while (current) {
		// We lock the current node ...
		omp_set_lock(&current->lock);
    
        if (current->val == val) {
			// We found the value in this node, we release its lock before to return
			omp_unset_lock(&current->lock);
            return index;
        }
        
        previous = current;
        current  = current->next;
        
        index++;
        
        // ... and we unlock it
		omp_unset_lock(&previous->lock);
    }

    return -1;
}

/*
 * This function deletes all element of a given linked list, and frees up all the allocated resources.
 */
void delete_list(node_t *head) {
    node_t *current = head;
    node_t *previous = NULL;
     
    // We lock the whole list
    while(current != NULL) {
		omp_set_lock(&current->lock);
		current = current->next;
	}
    
    // All nodes are locked, we can delete them
    current = head;
    while (current != NULL) {
        previous = current;
        current = current->next;
        
        // We can release and destroy the lock of the node before to free its allocated memory
        omp_unset_lock(&previous->lock);
        omp_destroy_lock(&previous->lock);
        free(previous);
        previous = NULL;
    }
}

int main(void) {
    
    node_t *test_list;
    
    #pragma omp parallel num_threads(4)
    {
		init_list(&test_list, 5); 
		
		int tid = omp_get_thread_num();
		printf("Thread %d\n", tid);
		
		for(int i = 0; i < 5; ++i) {
			add_first(&test_list, i);
		}
		
		
		//print_list(test_list);
		printf("Count 1 = %d\n", count(test_list));
		
		insert(&test_list, 3, 2);
		
		//print_list(test_list);
		printf("Count 2 = %d\n", count(test_list));
		
		append(test_list, 3);
		append(test_list, 4);
		append(test_list, 5);
		
		
		//print_list(test_list);
		printf("Count 3 = %d\n", count(test_list));
		
		pop(&test_list);
		
		remove_by_index(&test_list, 2);
		
		printf("Search for 5 -> index = %d\n", search_by_value(test_list, 5));
		
		//print_list(test_list);
		printf("Count 4 = %d\n", count(test_list));
		
		remove_by_value(&test_list, 5);
		printf("Search for 5 -> index = %d\n", search_by_value(test_list, 5));
		//print_list(test_list);
		printf("Count 5 = %d\n", count(test_list));
		
		
		delete_list(test_list);	
			
	}
	
    return 0;
}
