/**
 *  \file gdk_Cursor.cpp
 */

#include "gdk_Cursor.hpp"

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
#if 0 // todo
    { "new_from_pixmap",    &Cursor::new_from_pixmap },
    { "new_from_pixbuf",    &Cursor::new_from_pixbuf },
    { "new_from_name",      &Cursor::new_from_name },
    { "new_for_display",    &Cursor::new_for_display },
    { "get_display",        &Cursor::get_display },
    { "get_image",          &Cursor::get_image },
#endif
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

#if 0 // todo (GdkPixmap,GdkDisplay,..)
/*#
    @method new_from_pixmap
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

}


FALCON_FUNC Cursor::new_from_pixbuf( VMARG );
FALCON_FUNC Cursor::new_from_name( VMARG );
FALCON_FUNC Cursor::new_for_display( VMARG );
FALCON_FUNC Cursor::get_display( VMARG );
FALCON_FUNC Cursor::get_image( VMARG );
FALCON_FUNC Cursor::ref( VMARG );
FALCON_FUNC Cursor::unref( VMARG );
FALCON_FUNC Cursor::destroy( VMARG );
#endif

} // Gdk
} // Falcon
