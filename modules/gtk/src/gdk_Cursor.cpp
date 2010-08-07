/**
 *  \file gdk_Cursor.cpp
 */

#include "gdk_Cursor.hpp"

#include "gdk_Color.hpp"
#include "gdk_Display.hpp"
#include "gdk_Pixbuf.hpp"
#include "gdk_Pixmap.hpp"

#undef MYSELF
#define MYSELF Gdk::Cursor* self = Falcon::dyncast<Gdk::Cursor*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Cursor::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Cursor = mod->addClass( "GdkCursor", &Cursor::init );

    c_Cursor->setWKS( true );
    c_Cursor->getClassDef()->factory( &Cursor::factory );

    mod->addClassProperty( c_Cursor, "type" );

    Gtk::MethodTab methods[] =
    {
    { "new_from_pixmap",    &Cursor::new_from_pixmap },
    { "new_from_pixbuf",    &Cursor::new_from_pixbuf },
    { "new_from_name",      &Cursor::new_from_name },
    { "new_for_display",    &Cursor::new_for_display },
    { "get_display",        &Cursor::get_display },
    { "get_image",          &Cursor::get_image },
#if 0 // unused
    { "ref",                &Cursor::ref },
    { "unref",              &Cursor::unref },
    { "destroy",            &Cursor::destroy },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Cursor, meth->name, meth->cb );
}


Cursor::Cursor( const Falcon::CoreClass* gen, const GdkCursor* cursor )
    :
    Gtk::VoidObject( gen, cursor )
{
    incref();
}


Cursor::Cursor( const Cursor& other )
    :
    Gtk::VoidObject( other )
{
    incref();
}


Cursor::~Cursor()
{
    decref();
}


void Cursor::incref() const
{
    if ( m_obj )
        gdk_cursor_ref( (GdkCursor*) m_obj );
}


void Cursor::decref() const
{
    if ( m_obj )
        gdk_cursor_unref( (GdkCursor*) m_obj );
}


void Cursor::setObject( const void* cur )
{
    VoidObject::setObject( cur );
    incref();
}


bool Cursor::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkCursor* m_cursor = (GdkCursor*) m_obj;

    if ( s == "type" )
        it = (int64) m_cursor->type;
    else
        return defaultProperty( s, it );
    return true;
}


bool Cursor::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* Cursor::factory( const Falcon::CoreClass* gen, void* cursor, bool )
{
    return new Cursor( gen, (GdkCursor*) cursor );
}


/*#
    @class GdkCursor
    @brief Standard and pixmap cursors
    @param cursor_type cursor to create (GdkCursorType).

    These functions are used to create and destroy cursors. There is a number of
    standard cursors, but it is also possible to construct new cursors from
    pixmaps and pixbufs. There may be limitations as to what kinds of cursors
    can be constructed on a given display, see gdk_display_supports_cursor_alpha(),
    gdk_display_supports_cursor_color(), gdk_display_get_default_cursor_size()
    and gdk_display_get_maximal_cursor_size().

    Cursors by themselves are not very interesting, they must be be bound to a
    window for users to see them. This is done with gdk_window_set_cursor() or
    by setting the cursor member of the GdkWindowAttr struct passed to gdk_window_new().
 */
FALCON_FUNC Cursor::init( VMARG )
{
    Item* i_tp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tp || !i_tp->isInteger() )
        throw_inv_params( "GdkCursorType" );
#endif
    MYSELF;
    self->setObject( gdk_cursor_new( (GdkCursorType) i_tp->asInteger() ) );
}


/*#
    @method new_from_pixmap GdkCursor
    @brief Creates a new cursor from a given pixmap and mask.
    @param source the pixmap specifying the cursor (GdkPixmap).
    @param mask the pixmap specifying the mask, which must be the same size as source (GdkPixmap).
    @param fg the foreground color, used for the bits in the source which are 1 (GdkColor).
    @param bg the background color, used for the bits in the source which are 0 (GdkColor).
    @param x the horizontal offset of the 'hotspot' of the cursor.
    @param y the vertical offset of the 'hotspot' of the cursor.
    @return a new GdkCursor.

    Both the pixmap and mask must have a depth of 1 (i.e. each pixel has only 2
    values - on or off). The standard cursor size is 16 by 16 pixels. You can
    create a bitmap from inline data as in the below example.

    [...]
 */
FALCON_FUNC Cursor::new_from_pixmap( VMARG )
{
    Item* i_src = vm->param( 0 );
    Item* i_mask = vm->param( 1 );
    Item* i_fg = vm->param( 2 );
    Item* i_bg = vm->param( 3 );
    Item* i_x = vm->param( 4 );
    Item* i_y = vm->param( 5 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkPixmap )
        || !i_mask || !i_mask->isObject() || !IS_DERIVED( i_mask, GdkPixmap )
        || !i_fg || !i_fg->isObject() || !IS_DERIVED( i_fg, GdkColor )
        || !i_bg || !i_bg->isObject() || !IS_DERIVED( i_bg, GdkColor )
        || !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "GdkPixmap,GdkPixmap,GdkColor,GdkColor,I,I" );
#endif
    vm->retval( new Gdk::Cursor( vm->findWKI( "GdkCursor" )->asClass(),
        gdk_cursor_new_from_pixmap( GET_PIXMAP( *i_src ),
                                    GET_PIXMAP( *i_mask ),
                                    GET_COLOR( *i_fg ),
                                    GET_COLOR( *i_bg ),
                                    i_x->asInteger(),
                                    i_y->asInteger() ) ) );
}


/*#
    @method new_from_pixbuf GdkCursor
    @brief Creates a new cursor from a pixbuf.
    @param display the GdkDisplay for which the cursor will be created
    @param pixbuf the GdkPixbuf containing the cursor image
    @param x the horizontal offset of the 'hotspot' of the cursor.
    @param y the vertical offset of the 'hotspot' of the cursor.
    @return a new GdkCursor.

    Not all GDK backends support RGBA cursors. If they are not supported, a
    monochrome approximation will be displayed. The functions gdk_display_supports_cursor_alpha()
    and gdk_display_supports_cursor_color() can be used to determine whether
    RGBA cursors are supported; gdk_display_get_default_cursor_size() and
    gdk_display_get_maximal_cursor_size() give information about cursor sizes.

    On the X backend, support for RGBA cursors requires a sufficently new
    version of the X Render extension.
 */
FALCON_FUNC Cursor::new_from_pixbuf( VMARG )
{
    Item* i_display = vm->param( 0 );
    Item* i_pix = vm->param( 1 );
    Item* i_x = vm->param( 2 );
    Item* i_y = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_display || !i_display->isObject() || !IS_DERIVED( i_display, GdkDisplay )
        || !i_pix || !i_pix->isObject() || !IS_DERIVED( i_pix, GdkPixbuf )
        || !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "GdkDisplay,GdkPixbuf,I,I" );
#endif
    vm->retval( new Gdk::Cursor( vm->findWKI( "GdkCursor" )->asClass(),
                        gdk_cursor_new_from_pixbuf( GET_DISPLAY( *i_display ),
                                                    GET_PIXBUF( *i_pix ),
                                                    i_x->asInteger(),
                                                    i_y->asInteger() ) ) );
}


/*#
    @method new_from_name GdkCursor
    @brief Creates a new cursor by looking up name in the current cursor theme.
    @param name the name of the cursor
    @return a new GdkCursor, or NULL if there is no cursor with the given name
 */
FALCON_FUNC Cursor::new_from_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_name->asString() );
    GdkCursor* cur = gdk_cursor_new_from_name( GET_DISPLAY( vm->self() ), name.c_str() );
    if ( cur )
        vm->retval( new Gdk::Cursor( vm->findWKI( "GdkCursor" )->asClass(), cur ) );
    else
        vm->retnil();
}


/*#
    @method new_for_display GdkCursor
    @brief Creates a new cursor from the set of builtin cursors.
    @param display the GdkDisplay for which the cursor will be created
    @param cursor_type cursor to create (GdkCursorType)
 */
FALCON_FUNC Cursor::new_for_display( VMARG )
{
    Item* i_display = vm->param( 0 );
    Item* i_tp = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_display || !i_display->isObject() || !IS_DERIVED( i_display, GdkDisplay )
        || !i_tp || !i_tp->isInteger() )
        throw_inv_params( "GdkDisplay,GdkCursorType" );
#endif
    vm->retval( new Gdk::Cursor( vm->findWKI( "GdkCursor" )->asClass(),
                gdk_cursor_new_for_display( GET_DISPLAY( *i_display ),
                                            (GdkCursorType) i_tp->asInteger() ) ) );
}


/*#
    @method get_display GdkCursor
    @brief Returns the display on which the GdkCursor is defined.
    @return the GdkDisplay associated to cursor
 */
FALCON_FUNC Cursor::get_display( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Display( vm->findWKI( "GdkDisplay" )->asClass(),
                        gdk_cursor_get_display( GET_CURSOR( vm->self() ) ) ) );
}


/*#
    @method get_image GdkCursor
    @brief Returns a GdkPixbuf with the image used to display the cursor.
    @return a GdkPixbuf representing cursor, or NULL

    Note that depending on the capabilities of the windowing system and on the
    cursor, GDK may not be able to obtain the image data. In this case, NULL is returned.
 */
FALCON_FUNC Cursor::get_image( VMARG )
{
    NO_ARGS
    GdkPixbuf* pix = gdk_cursor_get_image( GET_CURSOR( vm->self() ) );
    if ( pix )
        vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), pix ) );
    else
        vm->retnil();
}


#if 0 // not used
FALCON_FUNC Cursor::ref( VMARG );
FALCON_FUNC Cursor::unref( VMARG );
FALCON_FUNC Cursor::destroy( VMARG );
#endif

} // Gdk
} // Falcon
