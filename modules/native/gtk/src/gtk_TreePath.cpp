/**
 *  \file gtk_TreePath.cpp
 */

#include "gtk_TreePath.hpp"

#undef MYSELF
#define MYSELF Gtk::TreePath* self = Falcon::dyncast<Gtk::TreePath*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreePath::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreePath = mod->addClass( "GtkTreePath", &TreePath::init );

    c_TreePath->setWKS( true );
    c_TreePath->getClassDef()->factory( &TreePath::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_from_string",        &TreePath::new_from_string },
#if 0 // unused
    { "new_from_indices",       &TreePath::new_from_indices },
#endif
    { "to_string",              &TreePath::to_string },
    { "new_first",              &TreePath::new_first },
    { "append_index",           &TreePath::append_index },
    { "prepend_index",          &TreePath::prepend_index },
    { "get_depth",              &TreePath::get_depth },
    { "get_indices",            &TreePath::get_indices },
#if 0 // unused
    { "get_indices_with_depth", &TreePath::get_indices_with_depth },
#endif
#if 0 // unused
    { "free",                   &TreePath::free },
#endif
    { "copy",                   &TreePath::copy },
    { "compare",                &TreePath::compare },
    { "next",                   &TreePath::next },
    { "prev",                   &TreePath::prev },
    { "up",                     &TreePath::up },
    { "down",                   &TreePath::down },
    { "is_ancestor",            &TreePath::is_ancestor },
    { "is_descendant",          &TreePath::is_descendant },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreePath, meth->name, meth->cb );
}


TreePath::TreePath( const Falcon::CoreClass* gen,
                    const GtkTreePath* path, const bool transfer )
    :
    Falcon::CoreObject( gen ),
    m_path( NULL )
{
    if ( path )
    {
        if ( transfer )
            m_path = (GtkTreePath*) path;
        else
            m_path = gtk_tree_path_copy( path );
    }
}


TreePath::~TreePath()
{
    if ( m_path )
        gtk_tree_path_free( m_path );
}


bool TreePath::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool TreePath::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* TreePath::factory( const Falcon::CoreClass* gen, void* path, bool )
{
    return new TreePath( gen, (GtkTreePath*) path );
}


void TreePath::setTreePath( const GtkTreePath* path, const bool transfer )
{
    assert( path && m_path == NULL );
    if ( transfer )
        m_path = (GtkTreePath*) path;
    else
        m_path = gtk_tree_path_copy( path );
}


/*#
    @class GtkTreePath
    @brief This structure refers to a row.
 */
FALCON_FUNC TreePath::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setTreePath( gtk_tree_path_new(), true );
}


/*#
    @method new_from_string GtkTreePath
    @brief Creates a new GtkTreePath initialized to path.
    @param path The string representation of a path.
    @return A newly-created GtkTreePath, or NULL

    path is expected to be a colon separated list of numbers. For example, the
    string "10:4:0" would create a path of depth 3 pointing to the 11th child of
    the root node, the 5th child of that 11th child, and the 1st child of that
    5th child. If an invalid path string is passed in, NULL is returned.
 */
FALCON_FUNC TreePath::new_from_string( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString path( i_path->asString() );
    GtkTreePath* tp = gtk_tree_path_new_from_string( path.c_str() );
    if ( tp )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                       tp,
                                       true ) );
    else
        vm->retnil();
}


#if 0 // unused
FALCON_FUNC TreePath::new_from_indices( VMARG )
#endif


/*#
    @method to_string GtkTreePath
    @brief Generates a string representation of the path.
    @return a string

    This string is a ':' separated list of numbers.
    For example, "4:10:0:3" would be an acceptable return value for this string.
 */
FALCON_FUNC TreePath::to_string( VMARG )
{
    NO_ARGS
    MYSELF;
    gchar* s = gtk_tree_path_to_string( self->getTreePath() );
    vm->retval( UTF8String( s ) );
    g_free( s );
}


/*#
    @method new_first GtkTreePath
    @brief Creates a new GtkTreePath. The string representation of this path is "0"
    @return a GtkTreePath
 */
FALCON_FUNC TreePath::new_first( VMARG )
{
    NO_ARGS
    vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                   gtk_tree_path_new_first(),
                                   true ) );
}


/*#
    @method append_index GtkTreePath
    @brief Appends a new index to a path.
    @param index The index.

    As a result, the depth of the path is increased.
 */
FALCON_FUNC TreePath::append_index( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    gtk_tree_path_append_index( self->getTreePath(), i_idx->asInteger() );
}


/*#
    @method prepend_index GtkTreePath
    @brief Prepends a new index to a path.
    @param index The index

    As a result, the depth of the path is increased.
 */
FALCON_FUNC TreePath::prepend_index( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    gtk_tree_path_prepend_index( self->getTreePath(), i_idx->asInteger() );
}


/*#
    @method get_depth GtkTreePath
    @brief Returns the current depth of path.
    @return The depth of path
 */
FALCON_FUNC TreePath::get_depth( VMARG )
{
    NO_ARGS
    MYSELF;
    vm->retval( gtk_tree_path_get_depth( self->getTreePath() ) );
}


/*#
    @method get_indices GtkTreePath
    @brief Returns the current indices of path.
    @return An array of the current indices, or NULL.

    This is an array of integers, each representing a node in a tree.
 */
FALCON_FUNC TreePath::get_indices( VMARG )
{
    NO_ARGS
    MYSELF;
    gint* indexes = gtk_tree_path_get_indices( self->getTreePath() );
    if ( indexes )
    {
        int i, cnt = 0;
        // tocheck: array supposedly terminated by -1, docs are missing for that
        for ( i = 0; indexes[i] != -1; ++i ) ++cnt;
        CoreArray* arr = new CoreArray( cnt );
        if ( cnt )
        {
            for ( i = 0; i < cnt; ++i )
                arr->append( indexes[i] );
        }
        vm->retval( arr );
    }
    else
        vm->retnil();
}


#if 0 // unused, see get_indices (arr len = depth)
FALCON_FUNC TreePath::get_indices_with_depth( VMARG )
#endif


#if 0 // unused
FALCON_FUNC TreePath::free( VMARG );
#endif


/*#
    @method copy GtkTreePath
    @brief Creates a new GtkTreePath as a copy of path.
    @return A new GtkTreePath.
 */
FALCON_FUNC TreePath::copy( VMARG )
{
    NO_ARGS
    MYSELF;
    vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                   gtk_tree_path_copy( self->getTreePath() ),
                                   true ) );
}


/*#
    @method compare GtkTreePath
    @brief Compares two paths.
    @param b A GtkTreePath to compare with.
    @return The relative positions of a and b

    If a appears before b in a tree, then -1 is returned.
    If b appears before a, then 1 is returned.
    If the two nodes are equal, then 0 is returned.
 */
FALCON_FUNC TreePath::compare( VMARG )
{
    Item* i_b = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_b || !i_b->isObject() || !IS_DERIVED( i_b, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* b = GET_TREEPATH( *i_b );
    MYSELF;
    vm->retval( gtk_tree_path_compare( self->getTreePath(), b ) );
}


/*#
    @method next GtkTreePath
    @brief Moves the path to point to the next node at the current depth.
 */
FALCON_FUNC TreePath::next( VMARG )
{
    NO_ARGS
    MYSELF;
    gtk_tree_path_next( self->getTreePath() );
}


/*#
    @method prev GtkTreePath
    @brief Moves the path to point to the previous node at the current depth, if it exists.
 */
FALCON_FUNC TreePath::prev( VMARG )
{
    NO_ARGS
    MYSELF;
    gtk_tree_path_prev( self->getTreePath() );
}


/*#
    @method up GtkTreePath
    @brief Moves the path to point to its parent node, if it has a parent.
    @return TRUE if path has a parent, and the move was made.
 */
FALCON_FUNC TreePath::up( VMARG )
{
    NO_ARGS
    MYSELF;
    vm->retval( (bool) gtk_tree_path_up( self->getTreePath() ) );
}

/*#
    @method down GtkTreePath
    @brief Moves path to point to the first child of the current path.
 */
FALCON_FUNC TreePath::down( VMARG )
{
    NO_ARGS
    MYSELF;
    gtk_tree_path_down( self->getTreePath() );
}


/*#
    @method is_ancestor GtkTreePath
    @brief Returns TRUE if descendant is a descendant of path.
    @param descendant another GtkTreePath
    @return TRUE if descendant is contained inside path
 */
FALCON_FUNC TreePath::is_ancestor( VMARG )
{
    Item* i_desc = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_desc || !i_desc->isObject() || !IS_DERIVED( i_desc, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* desc = GET_TREEPATH( *i_desc );
    MYSELF;
    vm->retval( (bool) gtk_tree_path_is_ancestor( self->getTreePath(), desc ) );
}


/*#
    @method is_descendant GtkTreePath
    @brief Returns TRUE if path is a descendant of ancestor.
    @param ancestor another GtkTreePath
    @return TRUE if ancestor contains path somewhere below it
 */
FALCON_FUNC TreePath::is_descendant( VMARG )
{
    Item* i_anc = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_anc || !i_anc->isObject() || !IS_DERIVED( i_anc, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* anc = GET_TREEPATH( *i_anc );
    MYSELF;
    vm->retval( (bool) gtk_tree_path_is_descendant( self->getTreePath(), anc ) );
}


} // Gtk
} // Falcon
