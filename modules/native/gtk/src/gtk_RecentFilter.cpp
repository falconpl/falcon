/**
 *  \file gtk_RecentFilter.cpp
 */

#include "gtk_RecentFilter.hpp"

#include "gtk_RecentFilterInfo.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RecentFilter::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RecentFilter = mod->addClass( "GtkRecentFilter", &RecentFilter::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_RecentFilter->getClassDef()->addInheritance( in );

    //c_RecentFilter->setWKS( true );
    c_RecentFilter->getClassDef()->factory( &RecentFilter::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_name",           RecentFilter::get_name },
    { "set_name",           RecentFilter::set_name },
    { "add_mime_type",      RecentFilter::add_mime_type },
    { "add_pattern",        RecentFilter::add_pattern },
    { "add_pixbuf_formats", RecentFilter::add_pixbuf_formats },
    { "add_application",    RecentFilter::add_application },
    { "add_group",          RecentFilter::add_group },
    { "add_age",            RecentFilter::add_age },
    { "add_custom",         RecentFilter::add_custom },
    { "get_needed",         RecentFilter::get_needed },
    { "filter",             RecentFilter::filter },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RecentFilter, meth->name, meth->cb );
}


RecentFilter::RecentFilter( const Falcon::CoreClass* gen, const GtkRecentFilter* filt )
    :
    Gtk::CoreGObject( gen, (GObject*) filt )
{}


Falcon::CoreObject* RecentFilter::factory( const Falcon::CoreClass* gen, void* filt, bool )
{
    return new RecentFilter( gen, (GtkRecentFilter*) filt );
}


/*#
    @class GtkRecentFilter
    @brief A filter for selecting a subset of recently used files

    A GtkRecentFilter can be used to restrict the files being shown in a
    GtkRecentChooser. Files can be filtered based on their name (with
    gtk_recent_filter_add_pattern()), on their mime type (with
    gtk_file_filter_add_mime_type()), on the application that has registered
    them (with gtk_recent_filter_add_application()), or by a custom filter
    function (with gtk_recent_filter_add_custom()).

    Filtering by mime type handles aliasing and subclassing of mime types; e.g.
    a filter for text/plain also matches a file with mime type application/rtf,
    since application/rtf is a subclass of text/plain. Note that GtkRecentFilter
    allows wildcards for the subtype of a mime type, so you can e.g. filter for image/*.

    Normally, filters are used by adding them to a GtkRecentChooser, see
    gtk_recent_chooser_add_filter(), but it is also possible to manually use a
    filter on a file with gtk_recent_filter_filter().
 */
FALCON_FUNC RecentFilter::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_recent_filter_new() );
}


/*#
    @method get_name GtkRecentFilter
    @brief Gets the human-readable name for the filter.
    @return the name of the filter, or NULL.
 */
FALCON_FUNC RecentFilter::get_name( VMARG )
{
    NO_ARGS
    vm->retval( UTF8String( gtk_recent_filter_get_name( GET_RECENTFILTER( vm->self() ) ) ) );
}


/*#
    @method set_name GtkRecentFilter
    @brief Sets the human-readable name of the filter.
    @param name the human readable name of filter

    This is the string that will be displayed in the recently used resources
    selector user interface if there is a selectable list of filters.
 */
FALCON_FUNC RecentFilter::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !( i_name->isNil() || i_name->isString() ) )
        throw_inv_params( "[S]" );
#endif
    if ( i_name->isNil() )
        gtk_recent_filter_set_name( GET_RECENTFILTER( vm->self() ), NULL );
    else
    {
        AutoCString name( i_name->asString() );
        gtk_recent_filter_set_name( GET_RECENTFILTER( vm->self() ), name.c_str() );
    }
}


/*#
    @method add_mime_type GtkRecentFilter
    @brief Adds a rule that allows resources based on their registered MIME type.
    @param mime_type a MIME type
 */
FALCON_FUNC RecentFilter::add_mime_type( VMARG )
{
    Item* i_tp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tp || !i_tp->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString tp( i_tp->asString() );
    gtk_recent_filter_add_mime_type( GET_RECENTFILTER( vm->self() ), tp.c_str() );
}


/*#
    @method add_pattern GtkRecentFilter
    @brief Adds a rule that allows resources based on a pattern matching their display name.
    @param pattern a file pattern
 */
FALCON_FUNC RecentFilter::add_pattern( VMARG )
{
    Item* i_pt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pt || !i_pt->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString pt( i_pt->asString() );
    gtk_recent_filter_add_pattern( GET_RECENTFILTER( vm->self() ), pt.c_str() );
}


/*#
    @method add_pixbuf_formats GtkRecentFilter
    @brief Adds a rule allowing image files in the formats supported by GdkPixbuf.
 */
FALCON_FUNC RecentFilter::add_pixbuf_formats( VMARG )
{
    NO_ARGS
    gtk_recent_filter_add_pixbuf_formats( GET_RECENTFILTER( vm->self() ) );
}


/*#
    @method add_application GtkRecentFilter
    @brief Adds a rule that allows resources based on the name of the application that has registered them.
    @param application an application name
 */
FALCON_FUNC RecentFilter::add_application( VMARG )
{
    Item* i_app = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_app || !i_app->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString app( i_app->asString() );
    gtk_recent_filter_add_application( GET_RECENTFILTER( vm->self() ), app.c_str() );
}


/*#
    @method add_group GtkRecentFilter
    @brief Adds a rule that allows resources based on the name of the group to which they belong
    @param group a group name
 */
FALCON_FUNC RecentFilter::add_group( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !i_grp->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString grp( i_grp->asString() );
    gtk_recent_filter_add_group( GET_RECENTFILTER( vm->self() ), grp.c_str() );
}


/*#
    @method add_age GtkRecentFilter
    @brief Adds a rule that allows resources based on their age - that is, the number of days elapsed since they were last modified.
    @param age number of days
 */
FALCON_FUNC RecentFilter::add_age( VMARG )
{
    Item* i_age = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_age || !i_age->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_recent_filter_add_age( GET_RECENTFILTER( vm->self() ), i_age->asInteger() );
}


/*#
    @method add_custom GtkRecentFilter
    @brief Adds a rule to a filter that allows resources based on a custom callback function.
    @param needed bitfield of flags (GtkRecentFilterFlags) indicating the information that the custom filter function needs.
    @param func callback function; if the function returns TRUE, then the file will be displayed.
    @param data data to pass to func, or nil

    The bitfield needed which is passed in provides information about what sorts
    of information that the filter function needs; this allows GTK+ to avoid
    retrieving expensive information when it isn't needed by the filter.
 */
FALCON_FUNC RecentFilter::add_custom( VMARG )
{
    Item* i_bit = vm->param( 0 );
    Item* i_func = vm->param( 1 );
    Item* i_data = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bit || !i_bit->isInteger()
        || !i_func || !i_func->isCallable()
        || !i_data )
        throw_inv_params( "GtkRecentFilterFlags,C,[X]" );
#endif
    MYSELF;
    GET_OBJ( self );
    g_object_set_data_full( (GObject*)_obj,
                            "__recent_filter_custom_func__",
                            new GarbageLock( *i_func ),
                            &CoreGObject::release_lock );
    g_object_set_data_full( (GObject*)_obj,
                            "__recent_filter_custom_func_data__",
                            new GarbageLock( *i_data ),
                            &CoreGObject::release_lock );
    gtk_recent_filter_add_custom( (GtkRecentFilter*)_obj,
                                  (GtkRecentFilterFlags) i_bit->asInteger(),
                                  &RecentFilter::exec_custom,
                                  (gpointer)_obj,
                                  NULL );
}


gboolean RecentFilter::exec_custom( const GtkRecentFilterInfo* info, gpointer filt )
{
    GarbageLock* func_lock = (GarbageLock*) g_object_get_data( (GObject*) filt,
                                        "__recent_filter_custom_func__" );
    GarbageLock* data_lock = (GarbageLock*) g_object_get_data( (GObject*) filt,
                                        "__recent_filter_custom_func_data__" );
    assert( func_lock && data_lock );
    Item func = func_lock->item();
    Item data = func_lock->item();
    VMachine* vm = VMachine::getCurrent(); // rather slow..
    vm->pushParam( new Gtk::RecentFilterInfo( vm->findWKI( "GtkRecentFilterInfo" )->asClass(), info ) );
    vm->pushParam( data );
    vm->callItem( func, 2 );
    Item it = vm->regA();
    if ( it.isBoolean() )
        return (gboolean) it.asBoolean();
    else
    {
        g_print( "RecentFilter::exec_custom: invalid callback (expected boolean)\n" );
        return FALSE;
    }
}


/*#
    @method get_needed GtkRecentFilter
    @brief Gets the fields that need to be filled in for the structure passed to gtk_recent_filter_filter()
    @return bitfield of flags indicating needed fields when calling gtk_recent_filter_filter()

    This function will not typically be used by applications; it is intended
    principally for use in the implementation of GtkRecentChooser.
 */
FALCON_FUNC RecentFilter::get_needed( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_recent_filter_get_needed( GET_RECENTFILTER( vm->self() ) ) );
}


/*#
    @method filter GtkRecentFilter
    @brief Tests whether a file should be displayed according to filter.
    @param filter_info a GtkRecentFilterInfo structure containing information about a recently used resource
    @return TRUE if the file should be displayed

    The GtkRecentFilterInfo structure filter_info should include the fields
    returned from gtk_recent_filter_get_needed().

    This function will not typically be used by applications; it is intended
    principally for use in the implementation of GtkRecentChooser.
 */
FALCON_FUNC RecentFilter::filter( VMARG )
{
    Item* i_info = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_info || !i_info->isObject() || !IS_DERIVED( i_info, GtkRecentFilterInfo ) )
        throw_inv_params( "GtkRecentfilterInfo" );
#endif
    vm->retval( (bool) gtk_recent_filter_filter( GET_RECENTFILTER( vm->self() ),
                                                 GET_RECENTFILTERINFO( *i_info ) ) );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
