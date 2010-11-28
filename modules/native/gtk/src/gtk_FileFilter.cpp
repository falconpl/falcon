/**
 *  \file gtk_FileFilter.cpp
 */

#include "gtk_FileFilter.hpp"

#include "gtk_FileFilterInfo.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void FileFilter::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_FileFilter = mod->addClass( "GtkFileFilter", &FileFilter::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_FileFilter->getClassDef()->addInheritance( in );

    //c_FileFilter->setWKS( true );
    c_FileFilter->getClassDef()->factory( &FileFilter::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_name",           &FileFilter::set_name },
    { "get_name",           &FileFilter::get_name },
    { "add_mime_type",      &FileFilter::add_mime_type },
    { "add_pattern",        &FileFilter::add_pattern },
    { "add_pixbuf_formats", &FileFilter::add_pixbuf_formats },
    { "add_custom",         &FileFilter::add_custom },
    { "get_needed",         &FileFilter::get_needed },
    { "filter",             &FileFilter::filter },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_FileFilter, meth->name, meth->cb );
}


FileFilter::FileFilter( const Falcon::CoreClass* gen, const GtkFileFilter* filt )
    :
    Gtk::CoreGObject( gen, (GObject*) filt )
{}


Falcon::CoreObject* FileFilter::factory( const Falcon::CoreClass* gen, void* filt, bool )
{
    return new FileFilter( gen, (GtkFileFilter*) filt );
}


/*#
    @class GtkFileFilter
    @brief A filter for selecting a file subset

    A GtkFileFilter can be used to restrict the files being shown in a
    GtkFileChooser. Files can be filtered based on their name (with
    gtk_file_filter_add_pattern()), on their mime type (with
    gtk_file_filter_add_mime_type()), or by a custom filter function (with
    gtk_file_filter_add_custom()).

    Filtering by mime types handles aliasing and subclassing of mime types; e.g.
    a filter for text/plain also matches a file with mime type application/rtf,
    since application/rtf is a subclass of text/plain. Note that GtkFileFilter
    allows wildcards for the subtype of a mime type, so you can e.g. filter
    for image/*.

    Normally, filters are used by adding them to a GtkFileChooser, see
    gtk_file_chooser_add_filter(), but it is also possible to manually use a
    filter on a file with gtk_file_filter_filter().

    Creates a new GtkFileFilter with no rules added to it. Such a filter doesn't
    accept any files, so is not particularly useful until you add rules with
    gtk_file_filter_add_mime_type(), gtk_file_filter_add_pattern(), or
    gtk_file_filter_add_custom().

    [...]
 */
FALCON_FUNC FileFilter::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_file_filter_new() );
}


/*#
    @method set_name GtkFileFilter
    @brief Sets the human-readable name of the filter.
    @param name the human-readable-name for the filter, or NULL to remove any existing name.

    This is the string that will be displayed in the file selector user
    interface if there is a selectable list of filters.
 */
FALCON_FUNC FileFilter::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !( i_name->isNil() || i_name->isString() ) )
        throw_inv_params( "[S]" );
#endif
    if ( i_name->isNil() )
        gtk_file_filter_set_name( GET_FILEFILTER( vm->self() ), NULL );
    else
    {
        AutoCString name( i_name->asString() );
        gtk_file_filter_set_name( GET_FILEFILTER( vm->self() ), name.c_str() );
    }
}


/*#
    @method get_name GtkFileFilter
    @brief Gets the human-readable name for the filter.
    @return The human-readable name of the filter, or NULL.
 */
FALCON_FUNC FileFilter::get_name( VMARG )
{
    NO_ARGS
    vm->retval( UTF8String( gtk_file_filter_get_name( GET_FILEFILTER( vm->self() ) ) ) );
}


/*#
    @method add_mime_type GtkFileFilter
    @brief Adds a rule allowing a given mime type to filter.
    @param mime_type name of a MIME type
 */
FALCON_FUNC FileFilter::add_mime_type( VMARG )
{
    Item* i_tp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tp || !i_tp->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString tp( i_tp->asString() );
    gtk_file_filter_add_mime_type( GET_FILEFILTER( vm->self() ), tp.c_str() );
}


/*#
    @method add_pattern GtkFileFilter
    @brief Adds a rule allowing a shell style glob to a filter.
    @param pattern a shell style glob
 */
FALCON_FUNC FileFilter::add_pattern( VMARG )
{
    Item* i_pt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pt || !i_pt->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString pt( i_pt->asString() );
    gtk_file_filter_add_pattern( GET_FILEFILTER( vm->self() ), pt.c_str() );
}


/*#
    @method add_pixbuf_formats GtkFileFilter
    @brief Adds a rule allowing image files in the formats supported by GdkPixbuf.
 */
FALCON_FUNC FileFilter::add_pixbuf_formats( VMARG )
{
    NO_ARGS
    gtk_file_filter_add_pixbuf_formats( GET_FILEFILTER( vm->self() ) );
}


/*#
    @method add_custom GtkFileFilter
    @brief Adds rule to a filter that allows files based on a custom callback function.
    @param bitfield bitfield of flags (GtkFileFilterFlags) indicating the information that the custom filter function needs.
    @param func callback function; if the function returns TRUE, then the file will be displayed.
    @param data data to pass to func, or nil

    The bitfield needed which is passed in provides information about what sorts
    of information that the filter function needs; this allows GTK+ to avoid
    retrieving expensive information when it isn't needed by the filter.
 */
FALCON_FUNC FileFilter::add_custom( VMARG )
{
    Item* i_bit = vm->param( 0 );
    Item* i_func = vm->param( 1 );
    Item* i_data = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bit || !i_bit->isInteger()
        || !i_func || !i_func->isCallable()
        || !i_data )
        throw_inv_params( "GtkFileFilterFlags,C,[X]" );
#endif
    MYSELF;
    GET_OBJ( self );
    g_object_set_data_full( (GObject*)_obj,
                            "__file_filter_custom_func__",
                            new GarbageLock( *i_func ),
                            &CoreGObject::release_lock );
    g_object_set_data_full( (GObject*)_obj,
                            "__file_filter_custom_func_data__",
                            new GarbageLock( *i_data ),
                            &CoreGObject::release_lock );
    gtk_file_filter_add_custom( (GtkFileFilter*)_obj,
                                (GtkFileFilterFlags) i_bit->asInteger(),
                                &FileFilter::exec_custom,
                                (gpointer)_obj,
                                NULL );
}


gboolean FileFilter::exec_custom( const GtkFileFilterInfo* info, gpointer filt )
{
    GarbageLock* func_lock = (GarbageLock*) g_object_get_data( (GObject*) filt,
                                        "__file_filter_custom_func__" );
    GarbageLock* data_lock = (GarbageLock*) g_object_get_data( (GObject*) filt,
                                        "__file_filter_custom_func_data__" );
    assert( func_lock && data_lock );
    Item func = func_lock->item();
    Item data = func_lock->item();
    VMachine* vm = VMachine::getCurrent(); // rather slow..
    vm->pushParam( new Gtk::FileFilterInfo( vm->findWKI( "GtkFileFilterInfo" )->asClass(), info ) );
    vm->pushParam( data );
    vm->callItem( func, 2 );
    Item it = vm->regA();
    if ( it.isBoolean() )
        return (gboolean) it.asBoolean();
    else
    {
        g_print( "FileFilter::exec_custom: invalid callback (expected boolean)\n" );
        return FALSE;
    }
}


/*#
    @method get_needed GtkFileFilter
    @brief Gets the fields that need to be filled in for the structure passed to gtk_file_filter_filter()
    @return bitfield of flags (GtkFileFilterFlags) indicating needed fields when calling gtk_file_filter_filter()

    This function will not typically be used by applications; it is intended
    principally for use in the implementation of GtkFileChooser.
 */
FALCON_FUNC FileFilter::get_needed( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_file_filter_get_needed( GET_FILEFILTER( vm->self() ) ) );
}


/*#
    @method filter GtkFileFilter
    @brief Tests whether a file should be displayed according to filter.
    @param filter_info a GtkFileFilterInfo structure containing information about a file.
    @return TRUE if the file should be displayed

    The GtkFileFilterInfo structure filter_info should include the fields
    returned from gtk_file_filter_get_needed().

    This function will not typically be used by applications; it is intended
    principally for use in the implementation of GtkFileChooser.
 */
FALCON_FUNC FileFilter::filter( VMARG )
{
    Item* i_info = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_info || !i_info->isObject() || !IS_DERIVED( i_info, GtkFileFilterInfo ) )
        throw_inv_params( "GtkFilefilterInfo" );
#endif
    vm->retval( (bool) gtk_file_filter_filter( GET_FILEFILTER( vm->self() ),
                                               GET_FILEFILTERINFO( *i_info ) ) );
}


} // Gtk
} // Falcon
