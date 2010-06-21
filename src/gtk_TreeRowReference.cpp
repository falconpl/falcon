/**
 *  \file gtk_TreeRowReference.cpp
 */

#include "gtk_TreeRowReference.hpp"

#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
#include "gtk_TreePath.hpp"

#undef MYSELF
#define MYSELF Gtk::TreeRowReference* self = Falcon::dyncast<Gtk::TreeRowReference*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeRowReference::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeRowReference = mod->addClass( "GtkTreeRowReference", &TreeRowReference::init );

    c_TreeRowReference->setWKS( true );
    c_TreeRowReference->getClassDef()->factory( &TreeRowReference::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_proxy",      &TreeRowReference::new_proxy },
    { "get_model",      &TreeRowReference::get_model },
    { "get_path",       &TreeRowReference::get_path },
    { "valid",          &TreeRowReference::valid },
#if 0 // unused
    { "free",           &TreeRowReference::free },
#endif
    { "copy",           &TreeRowReference::copy },
    { "inserted",       &TreeRowReference::inserted },
    { "deleted",        &TreeRowReference::deleted },
    { "reordered",      &TreeRowReference::reordered },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeRowReference, meth->name, meth->cb );
}


TreeRowReference::TreeRowReference( const Falcon::CoreClass* gen,
                    const GtkTreeRowReference* rowref, const bool transfer )
    :
    Falcon::CoreObject( gen ),
    m_rowref( NULL )
{
    if ( rowref )
    {
        if ( transfer )
            m_rowref = (GtkTreeRowReference*) rowref;
        else
            m_rowref = gtk_tree_row_reference_copy( (GtkTreeRowReference*) rowref );
    }
}


TreeRowReference::~TreeRowReference()
{
    if ( m_rowref )
        gtk_tree_row_reference_free( m_rowref );
}


bool TreeRowReference::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool TreeRowReference::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* TreeRowReference::factory( const Falcon::CoreClass* gen, void* rowref, bool )
{
    return new TreeRowReference( gen, (GtkTreeRowReference*) rowref );
}


void TreeRowReference::setTreeRowReference( const GtkTreeRowReference* rowref,
                                            const bool transfer )
{
    assert( rowref && m_rowref == NULL );
    if ( transfer )
        m_rowref = (GtkTreeRowReference*) rowref;
    else
        m_rowref = gtk_tree_row_reference_copy( (GtkTreeRowReference*) rowref );
}


/*#
    @class GtkTreeRowReference
    @brief Creates a row reference based on path.
    @param model A GtkTreeModel
    @param path A valid GtkTreePath to monitor
    @raise ParamError if the path is invalid

    This reference will keep pointing to the node pointed to by path, so long as
    it exists. It listens to all signals emitted by model, and updates its path
    appropriately. If path isn't a valid path in model, a param error is raised.
 */
FALCON_FUNC TreeRowReference::init( VMARG )
{
    Item* i_mdl = vm->param( 0 );
    Item* i_path = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !i_mdl->isObject() || !IS_DERIVED( i_mdl, GtkTreeModel )
        || !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreeModel,GtkTreePath" );
#endif
    GtkTreeModel* mdl = GET_TREEMODEL( *i_mdl );
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeRowReference* ref = gtk_tree_row_reference_new( mdl, path );
    if ( ref )
    {
        MYSELF;
        self->setTreeRowReference( ref );
    }
    else
        throw_inv_params( "Invalid GtkTreePath" ); // todo: translate
}


/*#
    @method new_proxy GtkTreeRowReference
    @brief You do not need to use this function. Creates a row reference based on path. This reference will keep pointing to the node pointed to by path, so long as it exists. If path isn't a valid path in model, then NULL is returned. However, unlike references created with gtk_tree_row_reference_new(), it does not listen to the model for changes. The creator of the row reference must do this explicitly using gtk_tree_row_reference_inserted(), gtk_tree_row_reference_deleted(), gtk_tree_row_reference_reordered().
    @param proxy A proxy GObject
    @param model A GtkTreeModel
    @param path A valid GtkTreePath to monitor
    @return A new GtkTreeRowReference
    @raise ParamError if the path is invalid

    These functions must be called exactly once per proxy when the corresponding
    signal on the model is emitted. This single call updates all row references
    for that proxy. Since built-in GTK+ objects like GtkTreeView already use this mechanism internally, using them as the proxy object will produce unpredictable results. Further more, passing the same object as model and proxy doesn't work for reasons of internal implementation.

    This type of row reference is primarily meant by structures that need to
    carefully monitor exactly when a row reference updates itself, and is not
    generally needed by most applications.
 */
FALCON_FUNC TreeRowReference::new_proxy( VMARG )
{
    Item* i_prox = vm->param( 0 );
    Item* i_mdl = vm->param( 1 );
    Item* i_path = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_prox || !i_prox->isObject() || !IS_DERIVED( i_prox, GObject )
        || !i_mdl || !i_mdl->isObject() || !IS_DERIVED( i_mdl, GtkTreeModel )
        || !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreeModel,GtkTreePath" );
#endif
    GObject* proxy = dyncast<Gtk::CoreGObject*>( i_prox->asObjectSafe() )->getGObject();
    GtkTreeModel* mdl = GET_TREEMODEL( *i_mdl );
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeRowReference* ref = gtk_tree_row_reference_new_proxy( proxy, mdl, path );
    if ( ref )
        vm->retval( new Gtk::TreeRowReference( vm->findWKI( "GtkTreeRowReference" )->asClass(),
                                               ref ) );
    else
        throw_inv_params( "Invalid GtkTreePath" );

}


/*#
    @method get_model GtkTreeRowReference
    @brief Returns the model that the row reference is monitoring.
    @return the model
 */
FALCON_FUNC TreeRowReference::get_model( VMARG )
{
    NO_ARGS
    MYSELF;
    GtkTreeModel* mdl = gtk_tree_row_reference_get_model( self->getTreeRowReference() );
    vm->retval( new Gtk::TreeModel( vm->findWKI( "GtkTreeModel" )->asClass(), mdl ) );
}


/*#
    @method get_path GtkTreeRowReference
    @brief Returns a path that the row reference currently points to, or NULL if the path pointed to is no longer valid.
    @return A current path, or NULL.
 */
FALCON_FUNC TreeRowReference::get_path( VMARG )
{
    NO_ARGS
    MYSELF;
    GtkTreePath* pth = gtk_tree_row_reference_get_path( self->getTreeRowReference() );
    if ( pth )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(), pth, true ) );
    else
        vm->retnil();
}


/*#
    @method valid GtkTreeRowReference
    @brief Returns TRUE if the reference refers to a current valid path.
    @return TRUE if reference points to a valid path.
 */
FALCON_FUNC TreeRowReference::valid( VMARG )
{
    NO_ARGS
    MYSELF;
    vm->retval( (bool) gtk_tree_row_reference_valid( self->getTreeRowReference() ) );
}


#if 0 // unused
FALCON_FUNC TreeRowReference::free( VMARG );
#endif


/*#
    @method copy GtkTreeRowReference
    @brief Copies a GtkTreeRowReference.
    @return a copy of reference.
 */
FALCON_FUNC TreeRowReference::copy( VMARG )
{
    NO_ARGS
    MYSELF;
    vm->retval( new Gtk::TreeRowReference( vm->findWKI( "GtkTreeRowreference" )->asClass(),
                gtk_tree_row_reference_copy( self->getTreeRowReference() ) ) );
}


/*#
    @method inserted GtkTreeRowReference
    @brief Lets a set of row reference created by gtk_tree_row_reference_new_proxy() know that the model emitted the "row_inserted" signal.
    @param proxy A GObject
    @param path The row position that was inserted
 */
FALCON_FUNC TreeRowReference::inserted( VMARG )
{
    Item* i_prox = vm->param( 0 );
    Item* i_path = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_prox || !i_prox->isObject() || !IS_DERIVED( i_prox, GObject )
        || !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GObject,GtkTreePath" );
#endif
    GObject* proxy = dyncast<Gtk::CoreGObject*>( i_prox->asObjectSafe() )->getGObject();
    GtkTreePath* path = GET_TREEPATH( *i_path );
    gtk_tree_row_reference_inserted( proxy, path );
}


/*#
    @method deleted GtkTreeRowReference
    @brief Lets a set of row reference created by gtk_tree_row_reference_new_proxy() know that the model emitted the "row_deleted" signal.
    @param proxy A GObject
    @param path The path position that was deleted
 */
FALCON_FUNC TreeRowReference::deleted( VMARG )
{
    Item* i_prox = vm->param( 0 );
    Item* i_path = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_prox || !i_prox->isObject() || !IS_DERIVED( i_prox, GObject )
        || !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GObject,GtkTreePath" );
#endif
    GObject* proxy = dyncast<Gtk::CoreGObject*>( i_prox->asObjectSafe() )->getGObject();
    GtkTreePath* path = GET_TREEPATH( *i_path );
    gtk_tree_row_reference_deleted( proxy, path );
}


/*#
    @method reordered GtkTreeRowReference
    @brief Lets a set of row reference created by gtk_tree_row_reference_new_proxy() know that the model emitted the "rows_reordered" signal.
    @param proxy A GObject
    @param path The parent path of the reordered signal
    @param iter The iter pointing to the parent of the reordered
    @param new_order The new order of rows (array of integers)
 */
FALCON_FUNC TreeRowReference::reordered( VMARG )
{
    Item* i_prox = vm->param( 0 );
    Item* i_path = vm->param( 1 );
    Item* i_iter = vm->param( 2 );
    Item* i_order = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_prox || !i_prox->isObject() || !IS_DERIVED( i_prox, GObject )
        || !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_order || !i_order->isArray() )
        throw_inv_params( "GObject,GtkTreePath,GtkTreeIter,A" );
#endif
    GObject* proxy = dyncast<Gtk::CoreGObject*>( i_prox->asObjectSafe() )->getGObject();
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    CoreArray* order = i_order->asArray();
    const int cnt = order->length();
    gint* norder = (gint*) memAlloc( sizeof( gint ) * cnt );
    Item it;
    for ( int i = 0; i < cnt; ++i )
    {
        it = order->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isInteger() )
        {
            memFree( norder );
            throw_inv_params( "I" );
        }
#endif
        norder[i] = it.asInteger();
    }
    gtk_tree_row_reference_reordered( proxy, path, iter, norder );
    memFree( norder );
}


} // Gtk
} // Falcon
