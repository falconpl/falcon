/**
 *  \file gtk_ImageMenuItem.cpp
 */

#include "gtk_ImageMenuItem.hpp"

#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ImageMenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ImageMenuItem = mod->addClass( "GtkImageMenuItem", &ImageMenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuItem" ) );
    c_ImageMenuItem->getClassDef()->addInheritance( in );

    c_ImageMenuItem->setWKS( true );
    c_ImageMenuItem->getClassDef()->factory( &ImageMenuItem::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_image",          &ImageMenuItem::set_image },
    { "get_image",          &ImageMenuItem::get_image },
    { "new_from_stock",     &ImageMenuItem::new_from_stock },
    { "new_with_label",     &ImageMenuItem::new_with_label },
    { "new_with_mnemonic",  &ImageMenuItem::new_with_mnemonic },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "get_use_stock",      &ImageMenuItem::get_use_stock },
    { "set_use_stock",      &ImageMenuItem::set_use_stock },
    { "get_always_show_image",&ImageMenuItem::get_always_show_image },
    { "set_always_show_image",&ImageMenuItem::set_always_show_image },
    //{ "set_accel_group",    &ImageMenuItem::set_accel_group },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ImageMenuItem, meth->name, meth->cb );
}


ImageMenuItem::ImageMenuItem( const Falcon::CoreClass* gen, const GtkImageMenuItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* ImageMenuItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new ImageMenuItem( gen, (GtkImageMenuItem*) itm );
}


/*#
    @class GtkImageMenuItem
    @brief A menu item with an icon

    A GtkImageMenuItem is a menu item which has an icon next to the text label.

    Note that the user can disable display of menu icons, so make sure to still
    fill in the text label.
 */
FALCON_FUNC ImageMenuItem::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_image_menu_item_new() );
}


/*#
    @method set_image GtkImageMenuItem
    @brief Sets the image of image_menu_item to the given widget.
    @param image a widget to set as the image for the menu item (or nil).

    Note that it depends on the show-menu-images setting whether the image will
    be displayed or not.
 */
FALCON_FUNC ImageMenuItem::set_image( VMARG )
{
    Item* i_img = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_img || !( i_img->isNil() || ( i_img->isObject()
        && IS_DERIVED( i_img, GtkWidget ) ) ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* img = i_img->isNil() ? NULL
                        : (GtkWidget*) COREGOBJECT( i_img )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_image_menu_item_set_image( (GtkImageMenuItem*)_obj, img );
}


/*#
    @method get_image GtkImageMenuItem
    @brief Gets the widget that is currently set as the image of image_menu_item.
    @return the widget set as image of image_menu_item.
 */
FALCON_FUNC ImageMenuItem::get_image( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_image_menu_item_get_image( (GtkImageMenuItem*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method new_from_stock GtkImageMenuItem
    @brief Creates a new GtkImageMenuItem containing the image and text from a stock item.
    @param stock_id the name of the stock item.
    @param accel_group TODO the GtkAccelGroup to add the menu items accelerator to, or NULL.
    @return a new GtkImageMenuItem.

    Some stock ids have preprocessor macros like GTK_STOCK_OK and GTK_STOCK_APPLY.

    If you want this menu item to have changeable accelerators, then pass in
    NULL for accel_group. Next call gtk_menu_item_set_accel_path() with an
    appropriate path for the menu item, use gtk_stock_lookup() to look up the
    standard accelerator for the stock item, and if one is found,
    call gtk_accel_map_add_entry() to register it.
 */
FALCON_FUNC ImageMenuItem::new_from_stock( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,[GtkAccelGroup]" );
    const gchar* id = args.getCString( 0 );
    GtkImageMenuItem* itm = (GtkImageMenuItem*) gtk_image_menu_item_new_from_stock( id, NULL );
    vm->retval( new Gtk::ImageMenuItem( vm->findWKI( "GtkImageMenuItem" )->asClass(), itm ) );
}


/*#
    @method new_with_label GtkImageMenuItem
    @brief Creates a new GtkImageMenuItem containing a label.
    @param label the text of the menu item.
    @return a new GtkImageMenuItem.
 */
FALCON_FUNC ImageMenuItem::new_with_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* lbl = args.getCString( 0 );
    GtkImageMenuItem* itm = (GtkImageMenuItem*) gtk_image_menu_item_new_with_label( lbl );
    vm->retval( new Gtk::ImageMenuItem( vm->findWKI( "GtkImageMenuItem" )->asClass(), itm ) );
}


/*#
    @method new_with_mnemonic GtkImageMenuItem
    @brief Creates a new GtkImageMenuItem containing a label.
    @param label the text of the menu item, with an underscore in front of the mnemonic character
    @return a new GtkImageMenuItem.

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the menu item.
 */
FALCON_FUNC ImageMenuItem::new_with_mnemonic( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* lbl = args.getCString( 0 );
    GtkImageMenuItem* itm = (GtkImageMenuItem*) gtk_image_menu_item_new_with_mnemonic( lbl );
    vm->retval( new Gtk::ImageMenuItem( vm->findWKI( "GtkImageMenuItem" )->asClass(), itm ) );
}


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method get_use_stock GtkImageMenuItem
    @brief Checks whether the label set in the menuitem is used as a stock id to select the stock item for the item.
    @return TRUE if the label set in the menuitem is used as a stock id to select the stock item for the item
 */
FALCON_FUNC ImageMenuItem::get_use_stock( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_image_menu_item_get_use_stock( (GtkImageMenuItem*)_obj ) );
}


/*#
    @method set_use_stock GtkImageMenuItem
    @brief If TRUE, the label set in the menuitem is used as a stock id to select the stock item for the item.
    @param use_stock TRUE if the menuitem should use a stock item
 */
FALCON_FUNC ImageMenuItem::set_use_stock( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_image_menu_item_set_use_stock( (GtkImageMenuItem*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_always_show_image GtkImageMenuItem
    @brief Returns whether the menu item will ignore the "gtk-menu-images" setting and always show the image, if available.
    @return TRUE if the menu item will always show the image
 */
FALCON_FUNC ImageMenuItem::get_always_show_image( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_image_menu_item_get_always_show_image( (GtkImageMenuItem*)_obj ) );
}


/*#
    @method set_always_show_image GtkImageMenuItem
    @brief If TRUE, the menu item will ignore the "gtk-menu-images" setting and always show the image, if available.
    @param always_show TRUE if the menuitem should always show the image

    Use this property if the menuitem would be useless or hard to use without the image.
 */
FALCON_FUNC ImageMenuItem::set_always_show_image( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_image_menu_item_set_always_show_image( (GtkImageMenuItem*)_obj,
                                               i_bool->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC ImageMenuItem::set_accel_group( VMARG );

#endif // GTK_CHECK_VERSION( 2, 16, 0 )


} // Gtk
} // Falcon
