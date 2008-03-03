/*
   Mini XML lib PLUS for C++

   Node::iterator class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_iterator.h,v 1.6 2005/02/24 13:39:22 jonnymind Exp $
*/

//TO BE INCLUDED IN NODE.H

//namespace MXML {

template<class __Node >
__iterator<__Node> &__iterator<__Node>::__prev()
{
	assert( this->m_node != m_base );

	if (this->m_node == 0 ) {
		if ( m_base->parent() != 0 )
			this->m_node = m_base->parent()->lastChild();
		else {
			this->m_node = m_base;
			while ( this->m_node->next() != 0 )
				this->m_node = this->m_node->next();
		}
	}
	else
		this->m_node = this->m_node->prev();
	return *this;
}

template< class __Node >
__iterator<__Node> &__iterator<__Node>::operator+=( int count )
{
	while ( count != 0 && this->m_node != 0 )
		operator++();
	return *this;
}

template< class __Node >
__iterator<__Node> &__iterator<__Node>::operator-=( int count )
{
	while ( count != 0 && this->m_node != m_base )
		operator--();
	return *this;
}

template< class __Node >
__iterator<__Node> __iterator<__Node>::operator+( int count )
{
	__iterator tmp = *this;
	while ( count > 0 && tmp.m_node != 0 ) {
		operator++();
		count--;
	}
	return tmp;
}

template< class __Node >
__iterator<__Node> __iterator<__Node>::operator-( int count )
{
	__iterator tmp = *this;
	while ( count > 0 && tmp.m_node != m_base ) {
		operator--();
		count--;
	}
	return tmp;
}

/*********************************************
* Deep iterator
*********************************************/
template< class __Node >
__iterator<__Node> &__deep_iterator<__Node>::__next()
{
	assert( this->m_node != 0 );

	if ( this->m_node->child() != 0 ) {
		this->m_node = this->m_node->child();
	}
	else if ( this->m_node->next() != 0 ) {
		this->m_node = this->m_node->next();
	}
	else {
		while ( this->m_node->parent() != 0 ) {
			this->m_node = this->m_node->parent();
			if ( this->m_node->next() != 0 )
				break;
		}
		this->m_node = this->m_node->next(); // can be NULL
	}

	return *this;
}

template< class __Node >
__iterator<__Node> &__deep_iterator<__Node>::__prev()
{
	return *this;
}

/***************************************************
* Find iterator
***************************************************/
template< class __Node >
__find_iterator<__Node>::__find_iterator( __Node *nd ):
__deep_iterator< __Node>( nd )
{
	m_maxmatch = 0;
}

template< class __Node >
__find_iterator<__Node>::__find_iterator( __Node *nd, Falcon::String name, Falcon::String attr,
	Falcon::String valatt, Falcon::String data):
__deep_iterator< __Node>( nd )
{
	m_name = name;
	m_attr = attr;
	m_valattr = valatt;
	m_data = data;
	m_maxmatch = 0;
	if ( m_name != "" ) m_maxmatch++;
	if ( m_attr !=  "" ) m_maxmatch++;
	if ( m_valattr != "" ) m_maxmatch++;
	if ( m_data != "" ) m_maxmatch++;
	__find();
};


template< class __Node >
__iterator<__Node> &__find_iterator<__Node>::__next()
{
	__deep_iterator<__Node>::__next();
	return __find();
}

template< class __Node >
__iterator<__Node> &__find_iterator<__Node>::__find()
{
	int matches;
	while ( this->m_node != 0 ) {
		matches = 0;
		if ( m_name != "" && m_name == this->m_node->name() )
			matches++;

		if ( m_attr != "" && this->m_node->hasAttribute( m_attr ) ) {
			matches++;
			if ( m_valattr != "" && this->m_node->getAttribute( m_attr ) == m_valattr )
				matches++;
		}

		if ( m_data != "" && this->m_node->data().find( m_data ) != Falcon::String::npos )
			matches++;

		if ( matches < m_maxmatch )
			__deep_iterator<__Node>::__next();
		else
			break;
	}

	return *this;
}

template< class __Node >
__iterator<__Node> &__find_iterator<__Node>::__prev()
{
	return *this;
}


/*********************************************
* Path Iterator
*********************************************/
template< class __Node >
__path_iterator<__Node>::__path_iterator( __Node *nd, const Falcon::String &path ):
__iterator< __Node>( nd )
{
	m_path = path;
	__find();
}

template< class __Node >
__iterator<__Node> &__path_iterator<__Node>::__next()
{
	__Node *ptr = this->m_node;
	Falcon::String name;
	Falcon::uint32 pos = m_path.rfind( "/" );

	if ( pos == Falcon::String::npos ) {
		pos = 0;
		name = m_path;
	}
	else {
		pos++;
		name = m_path.subString( pos );
	}

	this->m_node = this->m_node->next();   
   // todo: this sucks, must re-do it better
	while ( this->m_node ) {
		if ( name == "*" || this->m_node->name() == name ) break;
		this->m_node = this->m_node->next();   
	}

	return *this;
}

template< class __Node >
__iterator<__Node> &__path_iterator<__Node>::__prev()
{
	assert( this->m_node != 0 );

	__Node *ptr = this->m_node;
	this->m_node = this->m_node->prev();
   // todo: this sucks, must re-do it better
	while ( this->m_node != 0 ) {
		if ( this->m_node->name() == ptr->name() ) break;
		this->m_node = this->m_node->prev();
	}

	return *this;
}


template< class __Node >
__Node *__path_iterator<__Node>::subfind( __Node *parent, 
	Falcon::uint32 begin )
{
	Falcon::uint32 end = m_path.find( "/", begin );
	Falcon::String name = end == Falcon::String::npos? m_path.subString( begin ) : m_path.subString( begin, end );

	if ( name == "" ) return parent;

	parent = parent->child();
	while( parent != 0 ) {
		if ( name == "*"  || parent->name() == name ) {
			if ( end != Falcon::String::npos ) {
				parent = subfind( parent, end +1 );
			}
			break;
		}
		parent = parent->next();
	}

	return parent;
}

template< class __Node >
__iterator<__Node> &__path_iterator<__Node>::__find()
{
	if ( this->m_node == 0 ) return *this;

	__Node *rootNode = this->m_node;
	Falcon::String firstName;
	Falcon::uint32 pos;
	if ( rootNode->nodeType() == Node::typeDocument ) 
	{
		rootNode = rootNode->child();
		while( rootNode && rootNode->nodeType() != Node::typeTag )
			rootNode = rootNode->next();
		if ( ! rootNode )
		{
			this->m_node = 0;
			return *this;
		}
	}

	if ( m_path[0] == '/' ) 
	{         
		while( rootNode->parent() && rootNode->parent()->nodeType() != Node::typeDocument )
		{
			rootNode = rootNode->parent();
		}
		pos = m_path.find( "/", 1 );
		if( pos == Falcon::String::npos ) 
			firstName = m_path.subString( 1 );
		else
			firstName = m_path.subString( 1, pos );
	}
	else {
      // relative path, starting below us.
		rootNode = rootNode->child();
		pos = m_path.find( "/" );
		if( pos == Falcon::String::npos ) 
			firstName = m_path;
		else
			firstName = m_path.subString( 0, pos );
	}

	while ( rootNode != 0 ) 
	{
		if ( firstName == "*" || firstName == rootNode->name() ) 
		{
			if ( pos == Falcon::String::npos ) 
			{
				this->m_node = rootNode;
			}
			else {
				this->m_node = subfind( rootNode, pos+1 );
			}
			break;
		}
		rootNode = rootNode->next();
	}         

	return *this;
}

//}
/* end of mxml_iterator.h */
